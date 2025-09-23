from langchain_ollama.llms import OllamaLLM

# For prompts/templates
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableParallel

# For structured outputs
from pydantic import BaseModel, Field

class Notes(BaseModel) :
    KeyNotes: list[str]=Field(description='Summarize the content and put it in different points')

strparser = StrOutputParser()
pydanticparser = PydanticOutputParser(pydantic_object=Notes)

model = OllamaLLM(model='llama3.1:8b',num_predict=300)

instructions = pydanticparser.get_format_instructions()

prompt1 = PromptTemplate(
    template = 'Make a short summary of 5 lines of the following {content} given by user.',
    input_variables=['content'],
)

prompt2 = PromptTemplate(
    template="Make a questions and answer from this summary {content}.",
    input_variables=['summary'],
)

prompt3 = PromptTemplate(
    template="Merge the summary and questions/answers in a single notes/document -> summary: {summary}, questions/answer:{quiz}",
    input_variables=['summary','quiz'],
)

chain1 = RunnableParallel({
    'summary': prompt1 | model | strparser, # parellel chain
    'quiz': prompt2 | model | strparser
})

chain2 = prompt3 | model | strparser

chain3 = chain1 | chain2  # merging chains


text = """
Crypto.com is a cryptocurrency exchange company based in Singapore that offers various financial services, including an app, exchange, and noncustodial DeFi wallet, NFT marketplace, and direct payment service in cryptocurrency. As of June 2023, the company reportedly had 100 million customers and 4,000 employees.[2]

Crypto.com's user base increased from 10 million users in early 2021[3] to 100 million by mid-2024,[4] while its workforce exceeded 4,000 employees.[5] Regarding sponsorships and marketing activities, Crypto.com attracted actor Matt Damon as a brand ambassador,[6] collaborated with the soccer club Paris Saint-Germain F.C.,[7] and secured the naming rights for the Staples Center, now known as the Crypto.com Arena, in a 20-year agreement valued at $700 million.[8n North America, Crypto.com is licensed by Foreign MSB (Money Services Business) registrations with FINTRAC in Canada[61] and FinCEN in the United States for AML compliance. Additionally, the service holds Money Transmitter Licenses across various U.S. states, allowing it to operate as a payment and virtual asset service provider.[62]

In 2021, Crypto.com entered into a $216 million deal with IG Group, acquiring stakes in a US futures exchange and a binary trading group. This move aims to enable Crypto.com to offer derivatives and futures to US customers, an area often challenging for crypto exchanges due to the strict regulations surrounding these investment products. The acquisition includes the North American Derivatives Exchange (Nadex) and a 39% stake in Small Exchange, focusing on retail traders. This development is part of Crypto.com's broader strategy to comply with US regulations while expanding its service offerings.[63]

Australia
In December 2020, the company acquired an Australian Financial Service License by purchasing The Card Group Pty Ltd. This acquisition, approved by Australia’s Foreign Investment Review Board.[64]

The Australian Financial Service License enables Crypto.com to expand its services in the Australian market, adhering to the country's financial regulations. This move also facilitates the introduction of new offerings, such as launching its debit card in Australia.[65]

Crypto.com holds regulatory licenses, including an Australian Financial Services Licence (AFSL)[66] and an Australian Credit License (ACL) issued by the Australian Securities and Investments Commission (ASIC). The company is also registered with AUSTRAC for AML/CTF compliance, providing designated services such as issuing stored value cards, financial products, and operating as a digital currency exchange service provider.
"""

res = chain3.invoke({'content':text})
print(res, 'result')
