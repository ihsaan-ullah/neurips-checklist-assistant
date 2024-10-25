# Sample prompt
This is a sample prompt that we have used to get LLM feedback

```python
sample_prompt = """
You are provided with a “Paper” to be submitted to the NeurIPS conference. You are assisting the authors in preparing their “Answer” to one checklist “Question”. Please examine carefully the proposed author's “Answer” and the proposed author's “Justification” provided, and identify any discrepancies with the actual ”Paper” content, for this specific “Question”, taking into account the “Guidelines” provided to authors.  Afterwards, provide itemized, actionable feedback, based on the “Guidelines”, aiming to improve the paper quality. Concentrate on a few of the most significant improvements that can be made, and write in terse technical English. While Authors' Proposed Answer is generally preferred to be a "Yes", it is acceptable to answer "No" or "NA" provided a proper Authors' Proposed Justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). If the Authors' Proposed Answer is Yes, the Authors' Proposed Justification for the Answer should point to the section(s) within which related material for the question can be found. Note that the Authors' Proposed Justification is not expected to contain anything else (although it is fine if it contains more details). Finally, after performing all previous steps, conclude your review with a score for this specific “Question”, in a separate line (1: Everything OK or mild issues; 0.5: Needs improvements. Use this score sparingly; 0: Critical issues). Make sure that score is shown in a new line in this format “Score: score_value” and there is no content after the score.
Question: 
<START OF QUESTION>
{question}
<END OF QUESTION>
Answer: 
<START OF ANSWER>
{answer}
<END OF ANSWER>
Justification:
<START OF JUSTIFICATION>
{justification}
<END OF JUSTIFICATION>
Guidelines:
<START OF GUIDELINES>
{guidelines}
<END OF GUIDELINES>
Paper:
<START OF PAPER>
{paper}
<END OF PAPER>
"""
```