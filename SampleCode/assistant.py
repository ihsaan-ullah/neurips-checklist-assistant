# ------------------------------------------
# Imports
# ------------------------------------------
import os
import re
import fitz
import json
import warnings
import pandas as pd
from datetime import datetime as dt
from openai import OpenAI
warnings.filterwarnings("ignore")


# ------------------------------------------
# Assistant Class
# ------------------------------------------
class Assistant():

    def __init__(
        self,
        pdf_paper,
        checklist_data,
        PAPER_PROMPT,
        GPT_KEY,
        output_dir
    ):

        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.pdf_paper = pdf_paper
        self.checklist_data = checklist_data
        self.PAPER_PROMPT = PAPER_PROMPT
        self.GPT_KEY = GPT_KEY
        self.output_dir = output_dir

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def show_duration(self):
        print("\n---------------------------------")
        print(f'[✔] Total duration: {self.get_duration()}')
        print("---------------------------------")

    def set_directories(self):
        # Output data directory to write review result to
        # create output dir if it does not exist
        os.makedirs(self.output_dir, exist_ok=True)

    def clean(self, paper):

        # clean title
        paper["title"] = self.clean_title(paper["title"])

        # clean paper
        paper["paper"] = self.clean_paper(paper["paper"])

        # clean checklist
        paper["checklist"] = self.clean_checklist(paper["checklist"])

        return paper

    def clean_title(self, text):
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\-\s*\n', ' ', text)
        text = text.strip()
        return text

    def has_line_numbers(self, text):
        lines = text.split('\n')
        lines_to_iterate = len(lines) if len(lines) < 2000 else 2000
        numbered_lines = []
        for i in range(0, lines_to_iterate):
            numbered_lines.append(lines[i].strip().isdigit())
        if sum(numbered_lines) > int(lines_to_iterate * 0.10):
            return True
        return False

    def clean_paper(self, text):
        paper_has_line_numbers = self.has_line_numbers(text)
        if paper_has_line_numbers:
            text = re.sub(r'^(\d+)\n(\d+(\.\d+)?)\n(.+)$', r'\2 \4', text, flags=re.MULTILINE)
        else:
            text = re.sub(r'^(\d+)\n(.+)$', r'\1 \2', text, flags=re.MULTILINE)
        text = re.sub(r'\n\d+\n', r'\n', text)
        text = re.sub(r'\n\-\n', r'\n', text)
        text = re.sub(r'\-\s*\n', '', text)
        text = text.replace("’", "'")
        text = text.replace("\\'", "'")
        text = text.replace("- ", "")
        processed_text = ""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line.split()) < 6 and len(line.split()) > 1:
                processed_text += "\n"
                processed_text += line + "\n"
            else:
                processed_text += line
                processed_text += ' '
        text = processed_text.strip()
        return text

    def clean_checklist(self, text):
        text = re.sub(r'\n\d+', ' ', text)
        text = re.sub(r'\-\s*\n', '', text)
        text = re.sub(r'  . ', '\n', text)
        text = re.sub(r'([a-zA-Z]\.\d+)\n', r'\1 ', text)
        text = re.sub(r'([a-zA-Z])\n', r'\1 ', text)
        text = text.replace("’", "'")
        text = text.replace("\\'", "'")
        text = text.replace("- ", "")
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        text = text.replace("https://neurips.cc/public/ EthicsGuidelines", "https://neurips.cc/public/EthicsGuidelines")
        text = text.strip()
        return text

    def clean_guidelines(self, text):
        checklist_titles = [
            "Limitations",
            "Theory Assumptions and Proofs",
            "Experimental Result Reproducibility",
            "Open access to data and code",
            "Experimental Setting/Details",
            "Experiment Statistical Significance",
            "Experiments Compute Resources",
            "Code Of Ethics",
            "Broader Impacts",
            "Safeguards",
            "Licenses for existing assets",
            "New Assets",
            "Crowdsourcing and Research with Human Subjects",
            "Institutional Review Board (IRB) Approvals or Equivalent for Research with Human Subjects",
        ]
        for checklist_title in checklist_titles:
            text = text.replace(checklist_title, '')
        return text

    def get_paper_chunks(self, paper_text):

        try:
            # Identify main paper and appendices
            paper_end_index = paper_text.find("NeurIPS Paper Checklist")

            if paper_end_index == -1:
                raise ValueError("[-] Error: NeurIPS Paper Checklist not found. Please make sure that the checklist is at the end of the PDF.")

            paper = paper_text[:paper_end_index]

            total_allowed_words = 15000  # 400 per page (15 main pages, additional 20 pages)
            paper_words = paper.split()

            if len(paper_words) > total_allowed_words:
                paper_lines = paper.split('\n')
                paper_lines_to_keep = []
                word_count = 0
                for line in paper_lines:
                    line_words = line.split()
                    if word_count + len(line_words) <= total_allowed_words:
                        paper_lines_to_keep.append(line)
                        word_count += len(line_words)
                    else:
                        break

                paper = '\n'.join(paper_lines_to_keep)
                end_of_words_to_remove = ' '.join(paper.split()[-20:])
                print(f"[!] The paper is too long! Text after \"{end_of_words_to_remove}\" is removed and will not be considered for the LLM review.")

            # Identify checklist section
            checklist_start_index = paper_end_index
            checklist = paper_text[checklist_start_index:]

            # Identify title
            title_end_index = paper.find("Anonymous Author")
            if title_end_index == -1:
                title = paper.split("\n")[:2]
                title = ''.join(title)
            else:
                title = paper[:title_end_index]

            return {
                "title": title,
                "paper": paper,
                "checklist": checklist
            }
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"[-] Error occurred while extracting paper chunks in the {'paper' if not paper else 'checklist'} section: {e}")

    def get_pdf_text(self):
        paper_text = ""
        with fitz.open(self.pdf_paper) as doc:
            for page in doc:
                paper_text += page.get_text()
        return paper_text

    def parse_checklist(self, checklist):

        general_guidelines = """If you answer Yes to a question, in the justification please point to the section(s) where related material for the question can be found.
While "Yes" is generally preferable to "No", it is perfectly acceptable to answer "No" provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used").
"""
        checklist_df = pd.DataFrame(columns=['Question', 'Question_Title', 'Answer', 'Justification', 'Guidelines', 'Review', 'Score', 'LLM'])
        try:
            for question_index, checklist_item in enumerate(self.checklist_data):

                question_title = checklist_item["question_title"]
                question = checklist_item["question"]
                question_guidelines = checklist_item["guidelines"]

                question_regex = re.escape(question)
                pattern = re.compile(rf"Question:\s*{question_regex}(?:.*?Answer:\s*\[(.*?)\].*?Justification:\s*(.*?))(?:Guidelines:\s+(.*?))(?=Question:|\Z)", re.DOTALL)

                mtch = pattern.search(checklist)
                if mtch:
                    answer = mtch.group(1).strip()
                    justification = mtch.group(2).strip() if mtch.group(2).strip() else None

                    if justification is not None and justification.isdigit():
                        justification = None

                else:
                    answer, justification = "Not Found", "Not Found"

                temp_df = pd.DataFrame([
                    {
                        'Question': question,
                        'Question_Title': question_title,
                        'Answer': answer,
                        'Justification': justification,
                        'Guidelines': general_guidelines + question_guidelines}
                ])

                checklist_df = pd.concat([checklist_df, temp_df], ignore_index=True)
            return checklist_df

        except Exception as e:
            raise ValueError(f"[-] Error in extracting answers and justifications: {e}")

    def check_incomplete_questions(self, checklist_df):
        count_not_found = 0
        for i, row in checklist_df.iterrows():

            if row["Answer"] in ["TODO", "[TODO]"] or row["Justification"] in ["TODO", "[TODO]"] or row["Justification"] is None:
                print(f"[!] You haven't filled the answer or justificaiton for Question #: {i+1}")

            if row["Answer"] == "Not Found" or row["Justification"] == "Not Found":
                count_not_found += 1
                print(f"[!] There seems to be a problem with your answer or justificaiton for Question #: {i+1}. Please make sure that\n - you haven't changed the question in the checklist\n - you haven't removed the guidelines prodived for each question")

        if count_not_found == 15:
            raise ValueError("[-] All your answers or justifications are not found!. Please check that you have filled the checklist properly. If the problem is still there, please contact the organizers.")

    def get_LLM_feedback(self, paper, checklist_df):

        for index, row in checklist_df.iterrows():

            question_number = index + 1

            print(f"[*] Processing Question # {question_number}")

            q = row["Question"]
            a = row["Answer"]
            j = row["Justification"]
            g = row["Guidelines"]

            if a == "Not Found" or j == "Not Found":
                print(f"[!] Skipping Question # {question_number}. Answer or Justification for this question was not found!")
                checklist_df.loc[index, 'Review'] = "Answer or Justification for this question was not found!"
                checklist_df.loc[index, 'Score'] = 0
                continue

            paper_prompt = self.PAPER_PROMPT

            paper_prompt = paper_prompt.replace("{q}", q)
            paper_prompt = paper_prompt.replace("{a}", a)
            if j is None:
                j = ""
            paper_prompt = paper_prompt.replace("{j}", j)
            paper_prompt = paper_prompt.replace("{g}", g)
            paper_prompt = paper_prompt.replace("{paper}", paper)

            question_score, question_review, llm = self.get_single_question_LLM_feedback(paper_prompt)

            checklist_df.loc[index, 'Review'] = question_review
            checklist_df.loc[index, 'Score'] = question_score
            checklist_df.loc[index, 'LLM'] = llm

            print(f"[+] Question # {question_number}")

        return checklist_df

    def get_single_question_LLM_feedback(self, paper_prompt):

        max_tokens = 1000
        temperature = 1
        top_p = 1
        n = 1

        # Intialize GPT
        gpt_model = "gpt-4-turbo-preview"
        gpt_client = OpenAI(
            api_key=self.GPT_KEY,
        )

        number_of_times_question_processed = 0
        review_is_empty = True
        error_in_processing_question = False
        error_in_extracting_score = False

        score = -1
        llm_review = ""
        llm = ""

        while (review_is_empty or error_in_processing_question or error_in_extracting_score) and number_of_times_question_processed < 3:
            number_of_times_question_processed += 1

            if number_of_times_question_processed != 1:
                print("[!] Reprocessing this question!")

            # Get LLM review for a question
            try:

                llm = "GPT"
                user_prompt = {
                    "role": "user",
                    "content": paper_prompt
                }
                messages = [user_prompt]
                chat_completion = gpt_client.chat.completions.create(
                    model=gpt_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n
                )
                llm_review = chat_completion.choices[0].message.content
                error_in_processing_question = False
            except Exception as e:
                error_in_processing_question = True
                llm_review = f"Error: LLM failed to process this question!\n{str(e)}"
                score = -1
                print(f"[-] {llm_review}")
                continue

            # Check LLM review has no problem
            text = llm_review
            if len(text) > 50:
                review_is_empty = False
            else:
                review_is_empty = True
                llm_review = "Error: There seems to be a problem with this review! Empty/Truncated review"
                score = -1
                print(f"[-] {llm_review}")
                continue

            # Extract score from review
            try:
                score_pattern1 = r"Score:\s*([0-9]+(?:\.[0-9]+)?)"
                score_pattern2 = r"\*\*Score\*\*:\s*([0-9]+(?:\.[0-9]+)?)"
                match1 = re.search(score_pattern1, llm_review)
                match2 = re.search(score_pattern2, llm_review)
                if match1:
                    score = match1.group(1)
                    text = re.sub(r"Score:.*(\n|$)", "", text)
                elif match2:
                    score = match2.group(1)
                    text = re.sub(r"**Score**:.*(\n|$)", "", text)

                llm_review = text
                error_in_extracting_score = False

            except Exception as e:
                error_in_extracting_score = True
                llm_review = f"Error: Error: Unable to extract score from the LLM review!\n {str(e)}"
                score = -1
                print(f"[-] {llm_review}")
                continue

            return score, llm_review, llm

    def process_paper(self):

        # -----
        # Load text from PDF
        # -----
        print("[*] Loading and converting PDF to Text")
        paper_text = self.get_pdf_text()
        if paper_text == "":
            raise ValueError("[-] Reading PDF file failed or your PDF has no text! Please check that you have a valid PDF file.")
        print("[✔]")

        # -----
        # Get paper chunks
        # -----
        print("[*] Breaking down paper into chunks and cleaning text")
        self.paper = self.clean(self.get_paper_chunks(paper_text))
        print("[✔]")

        # -----
        # Parse Checklist
        # -----
        print("[*] Parsing checklist from text")
        self.paper["checklist_df"] = self.parse_checklist(self.paper["checklist"])
        print("[✔]")

        # -----
        # Incomplete answers
        # -----
        print("[*] Checking incomplete answers")
        self.check_incomplete_questions(self.paper["checklist_df"])
        print("[✔]")

        # -----
        # Get GPT Review
        # -----
        print("[*] Reviewing the paper's checklist")
        self.paper["checklist_df"] = self.get_LLM_feedback(self.paper["paper"], self.paper["checklist_df"])
        print("[✔]")

    def save(self):
        self.save_checklist()
        self.save_title()

    def save_checklist(self):
        print("[*] Saving checklists")
        checklist_file = os.path.join(self.output_dir, "paper_checklist.csv")
        self.paper["checklist_df"].replace('NA', 'Not Applicable', inplace=True)
        self.paper["checklist_df"].replace('N/A', 'Not Applicable', inplace=True)
        self.paper["checklist_df"].replace('[NA]', 'Not Applicable', inplace=True)
        self.paper["checklist_df"].replace('[N/A]', 'Not Applicable', inplace=True)
        self.paper["checklist_df"].to_csv(checklist_file, index=False)
        print("[✔]")

    def save_title(self):
        print("[*] Saving title")
        titles_dict = {"paper_title": self.paper["title"]}
        result_file = os.path.join(self.output_dir, "titles.json")
        with open(result_file, "w") as f_score:
            f_score.write(json.dumps(titles_dict, indent=4))
        print("[✔]")
