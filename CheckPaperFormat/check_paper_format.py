# ------------------------------------------
# Imports
# ------------------------------------------
import os
import re
import fitz
import argparse
import pandas as pd


def clean(paper):
    print("[*] Cleaning title, paper, checklist")
    # clean title
    paper["title"] = clean_title(paper["title"])
    # clean paper
    paper["paper"] = clean_paper(paper["paper"])
    # clean checklist
    paper["checklist"] = clean_checklist(paper["checklist"])
    print("[✔]")
    return paper


def clean_title(text):
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\-\s*\n', '', text)
    text = text.strip()
    return text


def clean_paper(text):
    text = re.sub(r'\n\d+', ' ', text)
    text = re.sub(r'\-\s*\n', '', text)
    text = re.sub(r'([a-zA-Z]\.\d+)\n', r'\1 ', text)
    text = re.sub(r'([a-zA-Z])\n', r'\1 ', text)
    text = text.replace("’", "'")
    text = text.replace("\\'", "'")
    text = text.replace("- ", "")
    processed_text = ""
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if len(line.split()) < 6:
            processed_text += '\n'
            processed_text += line + '\n'
        else:
            processed_text += line
            processed_text += ' '
    text = processed_text.strip()
    return text


def clean_checklist(text):
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
    text = text.strip()
    return text


def clean_guidelines(text):
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


def parse_checklist(checklist):

    print("[*] Parsing checklist")
    checklist_questions = [
        "Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?",
        "Does the paper discuss the limitations of the work performed by the authors?",
        "For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?",
        "Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?",
        "Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?",
        "Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?",
        "Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?",
        "For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?",
        "Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?",
        "Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?",
        "Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?",
        "Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?",
        "Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?",
        "For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?",
        "Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?"
    ]

    checklist_df = pd.DataFrame(columns=['Question', 'Answer', 'Justification', 'Guidelines'])
    try:
        for index, question in enumerate(checklist_questions):
            question_regex = re.escape(question)
            pattern = re.compile(rf"Question:\s+{question_regex}(?:.*?Answer:\s+\[(.*?)\].*?Justification:\s+(.*?))(?:Guidelines:\s+(.*?))(?=Question:|\Z)", re.DOTALL)

            mtch = pattern.search(checklist)
            if mtch:
                answer = mtch.group(1).strip()
                justification = mtch.group(2).strip() if mtch.group(2).strip() else None
                guidelines = mtch.group(3).strip() if mtch.group(3).strip() else None
                if guidelines:
                    guidelines = clean_guidelines(guidelines)

                if justification is not None and justification.isdigit():
                    justification = None

            else:
                answer, justification, guidelines = "Not Found", "Not Found", "Not Found"

            temp_df = pd.DataFrame([{'Question': question, 'Answer': answer, 'Justification': justification, 'Guidelines': guidelines}])
            checklist_df = pd.concat([checklist_df, temp_df], ignore_index=True)
            print(f"[+] Question # {index+1}")
        print("[✔]")
        return checklist_df

    except Exception as e:
        raise ValueError(f"[-] Error in extracting answers and justifications: {e}")


def check_incomplete_questions(checklist_df):
    print("[*] checking incomplete answers")
    for i, row in checklist_df.iterrows():
        if row["Answer"] in ["TODO", "[TODO]", "Not Found"] or row["Justification"] in ["TODO", "[TODO]", "Not Found"]:
            print(f"\t [!] There seems to be a problem with your answer or justificaiton for Question #: {i+1}")
    print("[✔]")


def get_pdf_text(pdf_file):
    print("[*] Extracting Text from PDF")
    try:
        pdf_file_path = os.path.join(pdf_file)
        paper_text = ""
        with fitz.open(pdf_file_path) as doc:
            for page in doc:
                paper_text += page.get_text()
        print("[✔]")
        return paper_text
    except Exception as e:
        raise ValueError(f"[-] Error in extracting answers and justifications: {e}")


def get_paper_chunks(paper_text):

    print("[*] Extracting paper chuncks: title, content, checklist")
    try:
        # Identify main paper and appendices
        paper_end_index = paper_text.find("NeurIPS Paper Checklist")

        if paper_end_index == -1:
            raise ValueError("[-] Error: NeurIPS Paper Checklist not found")

        paper = paper_text[:paper_end_index]

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

        print("[✔]")
        return {
            "title": title,
            "paper": paper,
            "checklist": checklist
        }
    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception(f"[-] Error occurred while extracting paper chunks in the {'paper' if not paper else 'checklist'} section: {e}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--paper', action="store", dest='paper', required=True,
                        help="Path to the PDF paper.")
    args = parser.parse_args()

    # Step 1: Extract text from PDF
    paper_text = get_pdf_text(args.paper)

    # Step 2: Divide paper into chunks: title, content, checklist
    paper_dict = get_paper_chunks(paper_text)

    # Step 3: Clean paper chunks
    paper_dict = clean(paper_dict)

    # Step 4: Parse checklist
    checklist_df = parse_checklist(paper_dict["checklist"])

    # Step 5: Check incomplete answers
    check_incomplete_questions(checklist_df)


# =======================================
# Example usage:
# python3 check_paper_format.py --paper ./paper.pdf
# =======================================
