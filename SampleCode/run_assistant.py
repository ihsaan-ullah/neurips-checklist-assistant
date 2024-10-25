# ------------------------------------------
# Imports
# ------------------------------------------
import argparse
from checklist_constants import checklist_data
from constants import prompt as PAPER_PROMPT
from constants import api_key as GPT_KEY
from assistant import Assistant


parser = argparse.ArgumentParser(description="Process a paper file.")
parser.add_argument('--paper', type=str, required=True, help="The PDF paper to process.")
args = parser.parse_args()



OUTPUT_DIR = "./Assistant_Result"


if __name__ == '__main__':

    print("############################################")
    print("### Starting Checklist Assistant")
    print("############################################\n")

    print("-"*50)
    print("## It may take 10-15 mins to process your paper.\n")
    print("-"*50)

    # Init Ingestion
    checklist_assistant = Assistant(
        pdf_paper=args.paper,
        checklist_data=checklist_data,
        PAPER_PROMPT=PAPER_PROMPT,
        GPT_KEY=GPT_KEY,
        output_dir=OUTPUT_DIR
    )

    checklist_assistant.set_directories()

    # Start timer
    checklist_assistant.start_timer()

    # process paper
    checklist_assistant.process_paper()

    # Save result
    checklist_assistant.save()

    # Stop timer
    checklist_assistant.stop_timer()

    # Show duration
    checklist_assistant.show_duration()

    print("\n----------------------------------------------")
    print("[✔] Checklist Assitant review completed successfully!")
    print("----------------------------------------------\n\n")