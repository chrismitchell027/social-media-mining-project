import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from more_itertools import chunked

# — CONFIGURATION —
INPUT_PATH = 'data/us_small_cleaned_tweets_with_labels_v_b64.csv'
OUTPUT_FMT = 'data/us_small_{topic}_cleaned_tweets_with_labels_v_b64.csv'
BATCH_SIZE = 16

labels = [
    "Strongly Opposes","Opposes","Weakly Opposes",
    "Neutral","Weakly Supports","Supports","Strongly Supports"
]

# Dictionary storing the political topic labels and their respective hypothesis templates
descriptions = {
    "Abortion": "This tweet {} abortion when it comes to pro-choice policies, such as supporting a woman’s right to choose, advocating for abortion access, or opposing restrictions on abortion.",
    "Voting Rights": "This tweet {} expanding voting rights, including supporting policies like automatic voter registration, increased access to polling stations, or fighting voter suppression.",
    "Taxes": "The tweet {} increasing taxes on the wealthy or corporations, including advocating for higher tax rates, closing tax loopholes, or promoting progressive taxation.",
    "Healthcare": "The tweet {} expanding access to healthcare or universal healthcare, such as supporting universal health insurance, affordable healthcare access, or opposing policies that limit healthcare coverage.",
    "Energy Policy": "This tweet {}  transitioning to renewable energy or green energy policies, including supporting measures to reduce reliance on fossil fuels, increasing investment in clean energy, or opposing environmental deregulation.",
    "Foreign Policy": "This tweet {} diplomatic engagement and peaceful conflict resolution, such as advocating for international cooperation, negotiation, or opposing military interventions and aggressive foreign policies.",
    "Criminal Justice": "This tweet {} criminal justice reform, including reducing incarceration, advocating for rehabilitation programs, decriminalizing certain offenses, or opposing harsh penalties and mass incarceration.",
    "Inflation": "This tweet {} actions to control inflation, including government intervention, tightening monetary policy, reducing government spending, or opposing inflationary measures and supporting tax cuts or monetary easing.",
    "Immigration": "This tweet {} more lenient immigration policies, including supporting refugee admissions, pathways to citizenship, opposing restrictive immigration laws, or advocating for comprehensive immigration reform.",
    "Gun Control": "This tweet {}  increased gun control measures, including supporting background checks, assault weapon bans, or gun buyback programs, or opposing policies that limit gun ownership rights.",
    "Education": "This tweet {} increasing funding for education, improving teacher pay, expanding access to higher education, or supporting policies that improve public education standards and accessibility.",
    "Unemployment": "This tweet {} increasing unemployment benefits, job creation programs, or other policies aimed at reducing unemployment, or opposing austerity measures that cut social safety nets and employment support.",
    "Social Security": "The tweet {}  expanding or preserving Social Security benefits, including supporting measures to increase the program’s funding, improve benefits, or oppose cuts to Social Security programs.",
    "LGBTQ Rights": "This tweet {} LGBTQ rights or policies protecting LGBTQ individuals, including supporting same-sex marriage, anti-discrimination laws, or opposing policies that limit the rights of LGBTQ people.",
    "Welfare": "This tweet {} expanding or maintaining government welfare programs, including supporting increased funding for social assistance programs, opposing cuts to welfare, or advocating for policies that reduce poverty.",
    "Minimum Wage": "This tweet {} raising the minimum wage, including supporting increases to the federal or state minimum wage, opposing efforts to reduce or eliminate wage increases, or advocating for better compensation for low-income workers.",
    "Climate Change": "This tweet {} taking action on climate change, including measures such as reducing greenhouse gas emissions, investing in renewable energy, or addressing environmental policies.",
    "Free Speech": "This tweet {} protecting free speech rights, including supporting laws that ensure the right to express opinions, even controversial ones, or opposing restrictions on speech in the media and public forums.",
    "Police Reform": "This tweet {} police reform, such as advocating for changes in policing practices, reducing police violence, improving accountability, or opposing policies that promote stricter law enforcement and harsher policing tactics.",
    "Student Loans": "This tweet {} policies related to student loans, including supporting loan forgiveness, reduced interest rates, or programs to ease student debt, or opposing proposals that may increase student loan burdens or limit forgiveness programs."
}

# If you've already done some topics:
completed_topics = {
    "Voting Rights"
}


classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0,                
    batch_size=BATCH_SIZE,   
    torch_dtype="auto"
)


df = pd.read_csv(INPUT_PATH)

# — PROCESS EACH TOPIC IN BATCHES —
for topic, hypothesis in descriptions.items():
    if topic in completed_topics:
        continue

    print(f"\nStarting topic: {topic}")
    df_topic = df[df['political_label'] == topic].copy()
    stances = []


    for batch in tqdm(
        chunked(df_topic['clean_tweet'], BATCH_SIZE),
        desc=f"Classifying {topic}",
        total=(len(df_topic) + BATCH_SIZE - 1)//BATCH_SIZE
    ):
        results = classifier(
            list(batch),
            candidate_labels=labels,
            hypothesis_template=hypothesis,
            multi_label=False
        )

        stances.extend([res['labels'][0] for res in results])


    df_topic['stance'] = stances
    df_topic.to_csv(OUTPUT_FMT.format(topic=(topic.replace(' ', '_'))), index=False)
    print(f"Finished {topic} — {len(stances)} tweets classified")

