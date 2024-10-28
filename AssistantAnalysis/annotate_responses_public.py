import pandas as pd
import zipfile
import pickle
import numpy as np
import re

# Replace with appropriate LLM keys to use
from openai import AzureOpenAI
from Azure.keys import KEY1, API_VERSION, ENDPOINT, DEPLOYMENT

# map long question names to short names
Q_MAP = {
    'Claims': 'Claims',
    'Limitations': 'Limitations',
    'Theoretical assumptions and proofs': 'Theory',
    'Experiments reproducibility': 'Reproducibility',
    'Code and data accessibility': 'Code and Data',
    'Experimental settings/details': 'Experimental Details',
    'Error bars': 'Error bars',
    'Compute resources': 'Compute resources',
    'NeurIPS code of ethics': 'Code of ethics',
    'Impacts': 'Impacts',
    'Safeguards': 'Safeguards',
    'Credits': 'Credits',
    'Documentation': 'Documentation',
    'Human subjects': 'Human subjects',
    'Risks': 'Risks',
}

### Data Loading Functions ###

def q_to_file(q):
    return q.lower().replace(' ', '_')

def extract_checklist(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open("paper_checklist.csv") as f:
            return pd.read_csv(f)

def extract_paper_parse(zip_path):
     with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open("article_dict.pickle") as f:
            return pickle.load(f)

def load_checklists(SUBMISSIONS_DATA_PATH='clean_data/submissions_final.csv'):
    df_sub = pd.read_csv(SUBMISSIONS_DATA_PATH)
    df_sub['zip_name'] = 'clean_data/'+ df_sub['zip_name']


    sub_ids = df_sub['submission_id']
    checklists = df_sub['zip_name'].apply(extract_checklist)
    df_checklists = pd.DataFrame()
    for sub_id, d in zip(sub_ids, checklists):
        d['submission_id'] = sub_id
        df_checklists = pd.concat([df_checklists, d])
    df_checklists = df_checklists.reset_index(drop=True)
    df_checklists['Question_Title_Short'] = df_checklists['Question_Title'].map(Q_MAP)
    return df_checklists

### Annotation Functions ###

# For a given piece of feedback on a given checklist question, summarize the feedback into a list of feedback types
def summarize_response(r):
        system_prompt = f'''
        You are a helfpful data annotater, clustering the types of feedback given on a scientific paper. 
        '''

        user_prompt = f'''
        In the following, you will be given feedback on a computer science paper submitted to a peer reviewed conference with a completed paper checklist. Please
        identify the main types of changes to the paper or paper checklist that the feedback suggests and summarize each point with a succinct name followed by a description.

        Provide the output in a comma-separated list in the following format:
        (Feedback type name): (Feedback type description) [newline]

        Here is the feedback given:\n
        ''' + r

        client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=KEY1,
            api_version=API_VERSION
        )

        response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        )

        out = response.choices[0].message.content
        return [s for s in out.split('\n') if len(s) > 0]

def summarize_responses(df_checklists, question):
    df_all_q = df_checklists[df_checklists['Question_Title_Short'] == question]
    return [summarize_response(r) for r in df_all_q['Review']]

# Processing block_sz responses at a time, annotate the responses to a given question to find the top {n_clusters} most common types of feedback
def get_top_clusters(df_checklists, question, n_clusters=3, block_sz=20):
    # get responses to the question and format as comma-separated list with index of each response before the response
    responses = df_checklists[df_checklists.Question_Title_Short == question].Review.reset_index(drop=True).values

    all_clusters = []

    # limit to 20 responses at a time by splitting responses into blocks of 20 at a time and annotating each block
    for i in range(0, len(responses), block_sz):
        responses_block = responses[i:i+block_sz]
        responses_str = ', '.join(responses_block)
        print(f'Analyzing responses {i+1}-{i+len(responses_block)} to the question: {question}.')

        system_prompt = f'''
        You are a helfpful data annotater, clustering the types of feedback given on a scientific paper. 
        '''

        user_prompt = f'''
        In the following, you will be given {block_sz} examples of feedback on a paper. The examples are provided as a comma-separated list. Please
        identify the {n_clusters} most frequent types of feedback given among the examples and cluster the examples into these types. For example, if many examples
        mention missing a required section, you might cluster these examples together as adding a section.

        Provide the output in a comma-separated list in the following format:
        Name: (Feedback type name); Description: (Feedback type description) [newline]

        Here is a comma-separated list of the feedback given:\n
        ''' + responses_str

        client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=KEY1,
            api_version=API_VERSION
        )

        response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        )
        
        top_clusters_str = response.choices[0].message.content
        top_clusters = top_clusters_str.split('\n')
        top_clusters = [c.split(';') for c in top_clusters]
        top_clusters = [c for c in top_clusters if len(c) > 1]

        all_clusters.extend(top_clusters)

    return all_clusters

# Using the top clusters found, annotate the responses to a given question to classify each response into one or more of the top clusters
def annotate_single_response(response, top_clusters):
    system_prompt = f'''
    You are a helfpful data annotater, who can classify whether feedback on a scientific paper covers a specific topic. 
    '''

    user_prompt = f'''
    In the following, you will be given a list of feedback categories and a specific feedback response. Please classify which of the categories
    the feedback response falls under. If the feedback response does not fall under any of the categories, please classify it as 'None'. If the feedback
    response falls under multiple categories, please include all the categories it covers.

    Provide the output in a semi-colon separated list of the categories the feedback response falls under. Do not include any other text in your response.
    
    Here is the list of feedback categories to use for the classification:\n
    ''' + '\n'.join([', '.join(c) for c in top_clusters]) + '\n\nHere is the feedback response to classify:\n' + response

    client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=KEY1,
            api_version=API_VERSION
        )

    response = client.chat.completions.create(
    model=DEPLOYMENT,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    )
    
    return response.choices[0].message.content    

def annotate_responses(df_checklists, top_clusters, question, lim_responses=None):
    responses_df = df_checklists[df_checklists.Question_Title_Short == question].reset_index(drop=True)

    responses = responses_df.Review.values
    ids = responses_df.submission_id.values

    if lim_responses:
        responses = responses[:lim_responses]

    annotations = []
    n = 0
    for response in responses:
        annotation = annotate_single_response(response, top_clusters)
        annotations.append(annotation)
        n += 1
        if n % 20 == 0:
            print(f'Annotated {n} responses')
    return zip(ids, annotations)

# Do a basic consolidation of the clusters by asking LLM to combine clusters with the same name and description
def consolidate_clusters(all_clusters, n_clusters=3):
    # consolidate clusters by asking LLM to combine clusters with the same name and description
    system_prompt = f'''
    You are a helfpful data annotater, who can consolidate different categories that are similar. 
    '''

    user_prompt = f'''
    In the following, you will be given a list of feedback categories. Some of the categories may be similar and can be combined.
    Please combine the categories that are similar based on name and description into as few distinct clusters as possible, but ensure that each category
    covers only one topic, not multiple unrelated topics. For example, if one category is
    'Clarification of Contribution and Scope' and another is 'Clarifying Contribution and Scope', these can be combined into a single category. Combine the names
    and descriptions of the categories that are similar into succinct single names and descriptions. However, if one topic is "Improve Limitaitons" and another is
    "Improve Experiments", these should not be combined as they cover different topics. Please consolidate into no more than {n_clusters} categories and choose the most common
    categories among the list.

    Provide the output in a comma-separated list in the following format:
    Name: (Feedback type name); Description: (Feedback type description) [newline]
    
    Here is the list of feedback categories to consolidate each separated by a newline:\n
    ''' + '\n'.join([', '.join(c) for c in all_clusters])

    client = AzureOpenAI(
            azure_endpoint=ENDPOINT,
            api_key=KEY1,
            api_version=API_VERSION
        )

    print(f'Consolidating {len(all_clusters)} clusters.')

    response = client.chat.completions.create(
    model=DEPLOYMENT,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    )
    
    top_clusters_str = response.choices[0].message.content
    top_clusters = top_clusters_str.split('\n')
    top_clusters = [c.split(';') for c in top_clusters]
    top_clusters = [c for c in top_clusters if len(c) > 1]

    print(f'Consolidated from {len(all_clusters)} to {len(top_clusters)} clusters.')
    # print out the top clusters found
    print('\n'.join([', '.join(c) for c in top_clusters]))

    return top_clusters

# Run the annotation pipeline for a given question to label each checklist with the main types of feedback given 
def run_annotation_pipeline(df_checklists, question, n_clusters=20, v=2):
    # Generate top types of feedback for a given question to use to annotate responses
    all_clusters = get_top_clusters(df_checklists, question, n_clusters)

    with open(f'annotations/all_clusters_{q_to_file(question)}_v{v}.pkl', 'wb') as f:
        pickle.dump(all_clusters, f)

    # Consolidate the clusters into the top n_clusters clusters to use for labelling
    top_clusters = consolidate_clusters(all_clusters, n_clusters=n_clusters)
    with open(f"annotations/top_clusters_{q_to_file(question)}_v{v}.pkl", 'wb') as f:
       pickle.dump(top_clusters, f)

    # Annotate the responses to the question with the top clusters found in previous steps
    annotations = annotate_responses(df_checklists, top_clusters, question)

    with open(f'annotations/llm_annotations_{q_to_file(question)}_v{v}.csv', 'w') as f:
        for id, annotation in annotations:
            f.write(f'{id}; {annotation}\n')

    return all_clusters, top_clusters, annotations

def load_annotations(question, v=2):
    with open(f'annotations/top_clusters_{q_to_file(question)}_v{v}.pkl', 'rb') as f:
        top_clusters = pickle.load(f)

    with open(f'annotations/llm_annotations_{q_to_file(question)}_v{v}.csv', 'r') as f:
        annotations = f.readlines()

    top_clusters = {c[0].replace('Name:', '').strip(): c[1].replace('Description:', '').strip() for c in top_clusters}
    
    # each annotation is submission id followed by up to 5 annotations separated
    annotations = [[s.strip() for s in a.split(';')] for a in annotations]
    df = pd.DataFrame(annotations)
    c_cols = [f'c{i+1}' for i in range(df.shape[1]-1)]
    df.columns = ['submission_id'] + c_cols
    df['clusters'] = df[c_cols].apply(lambda x: x.dropna().values, axis=1)
    df['submission_id'] = df['submission_id'].astype(int)
    df = df.drop(columns=c_cols)

    df_checklists = load_checklists()
    df_checklists = df_checklists[df_checklists.Question_Title_Short == question].reset_index(drop=True)[['submission_id', 'Review']]
    df = pd.merge(df, df_checklists, on='submission_id')

    return top_clusters, df

### Clustering of Feedback Annotations ### 

# edit distance between two strings
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i

    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]

def get_closest_cluster(c, clusters, max_dist=5):
    if c is None:
        return None
    if c in clusters:
        return c
    else:
        c_mod = min(clusters, key=lambda x: edit_distance(c, x))
        if edit_distance(c, c_mod) <= max_dist:
            return c_mod
        else:
            return None

# get top clusters from saved data
def get_top_clusters_data(annotations, clusters, num=3):
    all_clusters = np.hstack(annotations.clusters)
    # map each cluster to closest cluster in clusters by edit distance
    all_clusters = [get_closest_cluster(c, clusters) for c in all_clusters]
    all_clusters = [c for c in all_clusters if c is not None]

    cluster_names, counts = np.unique(all_clusters, return_counts=True)
    cluster_names = list(cluster_names)
    counts = list(counts)
    sorted_clusters = sorted(zip(cluster_names, counts), key=lambda x: x[1], reverse=True)

    n = annotations.shape[0]

    # list of name, frequency, description
    sorted_clusters = [(name, round(1.*cnt/n, 2), clusters[name])  for name,cnt in sorted_clusters]

    return sorted_clusters[:num]

# get all top 3 clusters for all questions
def analyze_top_clusters(qs):
    res = []

    for q in qs:
        print(f'Analyzing top clusters for question: {q}')
        clusters, annotations = load_annotations(q)
        top_clusters = get_top_clusters_data(annotations, clusters, num=20)
        d = pd.DataFrame(top_clusters, columns=['Feedback', 'Proportion of Responses', 'Description'])
        # set multi-level index for d to be question
        d['Question'] = q
        res.append(d)
    
    return pd.concat(res)

def parse_consolidated(consolidated):
    def parse_c(c):
        out = []
        for item in c:
            if len(item.split(':')) < 2:
                continue
            name, rest = item.split(':', 1)
            name = name.strip().replace('[', '').replace(']', '')
            if len(rest.split('[')) < 2:
                continue
            desc, lst = rest.split('[', 1)
            desc = desc.strip().replace(':', '').replace(';', '')
            lst = lst.replace(']', '').strip().split(',')
            lst = [l.strip() for l in lst]
            out.append((name, desc, lst))
        return out # list of tuples of the form (name, desc, consolidated names) 
    
    return [parse_c(c) for c in consolidated]

def load_summaries(question):
    with open(f'annotations/summaries/summaries_{q_to_file(question)}.pkl', 'rb') as f:
        summaries = pickle.load(f)
    summaries = [[p.replace('newline', '').replace('[', '').replace(']', '').strip() for p in s if p != 'newline' and len(p) > 0] for s in summaries]
    summaries = [[p.replace('(', '').replace(')', '').strip() for p in s if len(p) > 0] for s in summaries if len(s) > 0]
    summaries = [[re.sub(r'\d+\.', '', p).strip() for p in s] for s in summaries]
    summaries = [[re.sub(r'\d+', '', p).strip() for p in s] for s in summaries]
    return summaries

# Consolidate responses using LLM
def run_consolidate_responses(summaries):
     # consolidate feedback types by name only for feedback types that mean the exact same thing

    # unique_summaries = sorted(np.unique([item.split(':')[0] for sublist in summaries for item in sublist]))
    s = [item for sublist in summaries for item in sublist]
    all_summaries = (', ').join(s)
    system_prompt = f'''
        You are a helfpful data annotater, clustering the types of feedback given on a scientific paper. 
    '''

    user_prompt = f'''
        In the following, you will be given a list of {len(s)} types of feedback on computer science papers submitted to a peer reviewed conference. The feedback
        identifies the main types of changes to the paper or paper checklist with a succinct name followed by a description. Please combine the feedback into
        a single list of unique feedback types and provide a summary of the feedback types. If two feedback types are similar, but worded differently, please
        combine them into a single feedback type. If two feedback types are different, please keep them separate. You may return many feedback types, but please
        ensure that the feedback types are unique and do not overlap. Also, please ensure that the feedback type categories are not too general and keep different
        types of feedback in different categories. Below, you will find some examples of feedback types that should be combined and feedback types that should not be.
        I would expect there to be fewer feedback types than the original list of feedback types, but likely more than 5 feedback types.
         
        For example, the following pairs SHOULD be combined:
        
        "Clarity on Novel Contributions:..."
        "Highlight Novel Contributions:..."

        "Real-world Applicability Enhancement:..."
        "Linking Theory to Practice:..."

        "Clarification of Generalizability:..."
        "Generalizability Elaboration: Embed..."

        The following pairs SHOULD NOT be combined, keep them in separate categories:

        "Clarification of Claims:..."
        "Conciseness and Coherence:..."

        "Transparency in Methodology:..."
        "Transparency on Limitations:..."

        "Visual Reference Clarification:..."
        "Clarification of Novelty:..."

        Provide the output in a newline separated list in the following format:
        [Feedback type name]: [Feedback type description]; [list of feedback names consolidated] [newline]

        Make sure that the feedback type name is contained in square brackets, followed by a colon and a space, then the feedback type description, a semicolon, and a space, and finally a list of feedback names consolidated in square brackets. Each feedback type should be separated by a newline.

        Here is the feedback given:\n
        {all_summaries}
    '''
    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=KEY1,
        api_version=API_VERSION
    )

    response = client.chat.completions.create(
    model=DEPLOYMENT,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    )

    out = response.choices[0].message.content

    out = [l for l in out.split('\n') if len(l) > 0]
    print(f"Consolidated {len(s)} feedback types to {len(out)} feedback types.")

    return out

def consolidate_responses_step(summaries, block_size):
    out = []
    for i in range(0, len(summaries), block_size):
        print(f"Processing block {i} to {i+block_size}")
        block = summaries[i:i+block_size]
        r = run_consolidate_responses(block)
        out += [r]
        # break # remove this to run all blocks
    return out

# Consolidate the responses to a given question into a set of clusters in three hirarchical steps
def consolidate_responses(summaries):
    # consolidate with three merges
    c1 = consolidate_responses_step(summaries, 10)
    c2 = consolidate_responses_step([[s.split(';')[0] for s in c] for c in c1], len(c1) // 4)
    c3 = consolidate_responses_step([[s.split(':')[0] for s in c] for c in c2], len(c2))
    return c1, c2, c3

def get_closest_cluster_from_dict(c, cluster_dict):
    if c in cluster_dict.keys():
        return cluster_dict[c]
    else:
        c = get_closest_cluster(c, list(cluster_dict.keys()))
        if c is not None:
            return cluster_dict[c]
        else:
            return []

def get_cluster_mapping(summaries, c1, c2, c3):
    c1_map = {item[0]: item[2] for sublist in parse_consolidated(c1) for item in sublist}
    c2_map = {item[0]: item[2] for sublist in parse_consolidated(c2) for item in sublist}
    c3_map = {item[0]: item[2] for sublist in parse_consolidated(c3) for item in sublist}
    descriptions = {item[0]: item[1] for sublist in parse_consolidated(c3) for item in sublist}
    c_map = {}
    c_counts = {}

    cs_per_submission = [set([item.split(':')[0] for item in sublist]) for sublist in summaries]
    for category, subs in c3_map.items():
        s = []
        n = 0
        for s2 in subs:
            for s1 in get_closest_cluster_from_dict(s2, c2_map):
                s += get_closest_cluster_from_dict(s1, c1_map)
        
        for cs in cs_per_submission:
            if len(cs.intersection(s)) > 0:
                n += 1

        c_map[category] = s
        c_counts[category] = n
    
    # sort c_counts by value
    c_counts = {k: v for k, v in sorted(c_counts.items(), key=lambda item: item[1], reverse=True)}

    return c_map, c_counts, descriptions

def summarize_clusters(summaries, c1, c2, c3):
    c_map, c_counts, descriptions = get_cluster_mapping(summaries, c1, c2, c3)
    out = ""
    for category, n in c_counts.items():
        if n == 0:
            continue
        out += f"Category: {category}\n"
        out += f"Frequency: {n}\n"
        out += f"Description: {descriptions[category]}\n"
        # random sample of 5 feedback types
        sample = np.random.choice(c_map[category], min(n, 5), replace=False)
        out += f"Sample sub-categories of feedback: {sample}\n\n"
    return out

def main():
    df_checklists = load_checklists()
    for question in list(Q_MAP.values())[1:]:
        print(f'Generating summaries for question: {question}')
        # Generate summaries of the feedback for each question and each response
        summaries = summarize_responses(df_checklists, question)
        with open(f'annotations/summaries_{q_to_file(question)}.pkl', 'wb') as f:
            pickle.dump(summaries, f)
        # Use the feedback labels generated in the summary to annotate each response with the main types of feedback given
        run_annotation_pipeline(df_checklists, question)

    # Consolidate the responses to each question into a set of clusters using LLM to hierarchically merge clusters
    for question in list(Q_MAP.values()):
        summaries = load_summaries(question)
        c1, c2, c3 = consolidate_responses(summaries)
        with open(f'annotations/consolidated_{q_to_file(question)}.pkl', 'wb') as f:
            pickle.dump([c1, c2, c3], f)

if __name__ == '__main__':
    main()