import openai
import dotenv
import os
from enum import Enum


class GPTVersion(str, Enum):
    NONE = "none"
    GPT3_5 = "GPT 3.5 Turbo"
    GPT4 = "GPT 4"


def load_openai_key(path):
    dotenv.load_dotenv(path)
    openai.api_key = os.getenv("OPENAI_API_KEY")


def build_message_list_explain_equation(query: str):
    return [
        {
            "role": "system",
            "content": """You are an intelligent and knowledgeable AI chatbot specializing in biology and molecular sciences. Your purpose is to provide insightful explanations, answer questions, and assist users in understanding complex biological concepts.
            You are proficient in providing concise definitions, explaining biological processes, describing experimental techniques, and elucidating the interconnections between different biological phenomena""",
        },
        {
            "role": "user",
            "content": """
                What does this mean analogically? I found this by doing equations with vector embeddings and I'm trying to reason why this makes sense.

                ProteinMutation_p_R199Q_RS_200850756 (aka 'uncharacterized LOC124903011' or  'D-amino acid oxidase')   -    Chemical_MESH_C000334 (aka butanediol divinylether)   +    Gene_492771 (aka 8 subunit of ESCRT-II) =     SNP_rs2284018 (aka 'calcium voltage-gated channel auxiliary subunit gamma 2')

                Really try to think outside the box to find why this could be reasonable. Use this as a generative way to help think of biological hypotheses.
            """,
        },
        {
            "role": "assistant",
            "content": """
            ProteinMutation_p_R199Q_RS_200850756: This refers to a specific mutation in the protein D-amino acid oxidase (DAO). DAO is an enzyme that metabolizes D-amino acids, including D-serine, which plays a role in signal transmission in the brain.

            Chemical_MESH_C000334 (butanediol divinylether): This is a chemical compound, specifically an ether, often used to make polymeric membranes and microspheres.

            Gene_492771 (8 subunit of ESCRT-II): ESCRT-II is part of the Endosomal Sorting Complex Required for Transport, which helps in the transport and sorting of proteins within cells.

            SNP_rs2284018 (calcium voltage-gated channel auxiliary subunit gamma 2): This represents a single nucleotide polymorphism (SNP), or a small genetic change, in the gene encoding for a component of a calcium channel, which play a crucial role in cell signaling.

            The subtraction of butanediol divinylether from the protein mutation might represent removing the effect or influence of this chemical on the protein. Then, adding the 8 subunit of ESCRT-II may represent the introduction of a new effect or influence on the DAO protein mutation due to this gene. The result is SNP_rs2284018, a SNP associated with a calcium channel subunit.

            One hypothesis, then, could be that the protein mutation in DAO, when not influenced by the butanediol divinylether microspheres, but instead affected by the ESCRT-II gene, could result in changes that are similar to the SNP in the calcium channel auxiliary subunit gamma 2 (a change in calcium or in voltage).

            To further clarify, maybe the original protein mutation has a certain effect on cell signaling, perhaps through D-serine metabolism. Removing the influence of butanediol divinylether and introducing the influence of ESCRT-II could change how this protein mutation affects cell signaling, resulting in an effect similar to the SNP in the calcium channel gene.
            """,
        },
        {
            "role": "user",
            "content": f"""
        What does this mean analogically? I found this by doing equations with vector embeddings and I'm trying to reason why this makes sense.

        {query}

        Really try to think outside the box to find why this could be reasonable. Use this as a generative way to help think of biological hypotheses.
        """,
        },
    ]


def gpt_rationale(query: str, gpt_version: GPTVersion):
    completion = openai.ChatCompletion.create(
        model=gpt_version.value, messages=build_message_list_explain_equation(query)
    )
    feature_string = completion.choices[0].message.content
    return feature_string
