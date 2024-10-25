# ------------------------------------------
# Imports
# ------------------------------------------
import os
import re
import sys
import json
import base64
import pandas as pd
from datetime import datetime as dt
from jinja2 import Template


class ResultCompiler:
    def __init__(
        self,
        result_dir
    ):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.scores_dict = {}
        self.result_dir = result_dir

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
        print(f"[✔] Total duration: {self.get_duration()}")
        print("---------------------------------")

    def set_directories(self):

        score_file_name = "scores.json"
        html_file_name = "detailed_results.html"

        # score file to write score into
        self.score_file = os.path.join(self.result_dir, score_file_name)
        # html file to write score and figures into
        self.html_file = os.path.join(self.result_dir, html_file_name)
        # html template firle
        self.html_template_file = "template.html"

    def read_csv(self, csv):
        df = pd.read_csv(csv)
        df.replace('Not Applicable', 'NA', inplace=True)
        return df

    def load_ingestion_result(self):
        print("[*] Reading checklist assistant reviews")

        self.paper = None

        csv_file = os.path.join(self.result_dir, "paper_checklist.csv")
        titles_file = os.path.join(self.result_dir, "titles.json")

        # load titles file
        with open(titles_file) as f:
            titles = json.load(f)

        if os.path.exists(csv_file):
            timestamp = dt.now().strftime("%Y-%m-%d %H:%M:%S")
            title_to_be_encoded = f"{titles['paper_title']} - {timestamp}"
            encoded_title = base64.b64encode(title_to_be_encoded.encode()).decode('utf-8')
            self.paper = {
                "checklist_df": self.read_csv(csv_file),
                "title": titles["paper_title"],
                "encoded_title": encoded_title
            }
        else:
            raise ValueError("[-] Checklist CSV not found!")

        print("[✔]")

    def convert_text_to_html(self, text):
        try:
            html_output = ""

            # Split the text into lines
            lines = text.split('\n')

            # Iterate through each line in the text
            for line in lines:

                line = line.replace("```plaintext", "")
                line = line.replace("```", "")

                if line.strip() in ["**", "#", "##", "###", "# Score", "## Score", "### Score"]:
                    continue

                if '**' in line:
                    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)

                if len(line) == 0:
                    html_output += "<br>"
                elif line.startswith('# '):
                    html_output += f"<h3>{line.strip()[2:]}</h3>"
                elif line.startswith('## '):
                    html_output += f"<h3>{line.strip()[3:]}</h3>"
                elif line.startswith('### '):
                    html_output += f"<h3>{line.strip()[4:]}</h3>"
                elif line.startswith('#### '):
                    html_output += f"<h4>{line.strip()[5:]}</h4>"
                elif line.startswith('- ') or line.startswith('* '):
                    html_output += f"○ &nbsp;&nbsp; {line.strip()[2:]}<br>"
                elif re.match(r'^\d+\.', line.strip()):
                    html_output += f"○ &nbsp;&nbsp; {line.strip()[2:]}<br>"
                elif line.startswith("    *") or line.startswith("   -") or line.startswith("    -"):
                    nested_line_text = line
                    nested_line_text = nested_line_text.replace("    *", "")
                    nested_line_text = nested_line_text.replace("   -", "")
                    nested_line_text = nested_line_text.replace("    -", "")
                    html_output += f"&nbsp;&nbsp;&nbsp; • &nbsp;&nbsp; {nested_line_text}<br>"
                elif line.startswith("        *") or line.startswith("        -"):
                    nested_line_text = line
                    nested_line_text = nested_line_text.replace("        *", "")
                    nested_line_text = nested_line_text.replace("        -", "")
                    html_output += f"&nbsp;&nbsp;&nbsp;&nbsp; ○ &nbsp;&nbsp; {nested_line_text}<br>"
                elif line.startswith("   "):
                    html_output += f"&nbsp;&nbsp;&nbsp;{line.strip()}<br>"
                else:
                    html_output += f"{line.strip()}<br>"

            return html_output
        except:
            return text

    def write_detailed_results(self):
        print("[*] Writing detailed result")

        with open(self.html_template_file) as file:
            template_content = file.read()

        template = Template(template_content)

        # Prepare data
        paper_dict_for_template = {
            "title": self.paper["title"]
        }
        reviews = []
        for index, row in self.paper["checklist_df"].iterrows():

            question_number = index + 1
            reviews.append({
                "question_no": question_number,
                "question_id": f"question-{question_number}",
                "question": row['Question'],
                "question_title": row["Question_Title"],
                "answer": row['Answer'],
                "justification": row['Justification'],
                "review": self.convert_text_to_html(row['Review']),
                "score": row['Score']
            })
        paper_dict_for_template["reviews"] = reviews

        data = {
            "paper": paper_dict_for_template,
        }

        rendered_html = template.render(data)

        with open(self.html_file, 'w', encoding="utf-8") as f:
            f.write(rendered_html)

        print("[✔]")
