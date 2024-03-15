import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv

load_dotenv()
# %%
from pypdf import PdfReader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# %%
import os
import getpass

# %% [markdown]
# PDF parsing: Extracting meaningful content while ignoring less useful information like empty lines, headers, and footers using visitor_text:

# %%
document_content = None

def visitor_body(text, cm, tm, fontDict, fontSize):
    y = tm[5]
    if text and 35 < y < 770:
        page_contents.append(text)

with open(f'/home/sysadmin/llmops-course/AIE/Final-Project/datasets/pdf/source/ZMP_55852_XBO_1000_W_HS_OFR.pdf', 'rb') as file:
    pdf_reader = PdfReader(file)

    page_contents = []

    for page in pdf_reader.pages:
        page.extract_text(visitor_text=visitor_body)

    document_content = "\n".join(page_contents)

print(document_content)

# %%
system_message = """You analyze product descriptions to export them into a JSON format. I will present you with a product data sheet and describe the individual JSON objects and properties with <<<. You then create a JSON object from another product data sheet.

>>> Example product:

Product family benefits <<< benefits (string[])
_
Short arc with very high luminance for brighter screen illumination <<< benefits.[*]
_
Constant color temperature of 6,000 K throughout the entire lamp lifetime <<< benefits.[*]

[..]

_
Wide dimming range <<< benefits.[*]
Product family features <<< product_family (object)
_
Color temperature: approx. 6,000 K (Daylight) <<< product_family.temperature = 6000
_
Wattage: 450…10,000 W <<< product_family.watts_min = 450, product_family.watts_max = 10000
_
Very good color rendering index: Ra >
Product datasheet



XBO 1000 W/HS OFR <<< name
XBO for cinema projection | Xenon short-arc lamps 450…10,000 W <<< description

[..]

Technical data
Electrical data <<< technical_data (object)
Nominal current
50 A <<< technical_data.nominal_current = 50.00
Current control range
30…55 A <<< technical_data.control_range = 30, technical_data.control_range = 55
Nominal wattage
1000.00 W <<< technical_data.nominal_wattage = 1000.00
Nominal voltage
19.0 V <<< technical_data.nominal_voltage = 19.0
Dimensions & weight <<< dimensions (object)

[..]

Safe Use Instruction
The identification of the Candidate List substance is <<< environmental_information.safe_use (beginning of string)

sufficient to allow safe use of the article. <<< environmental_information.safe_use (end of string)
Declaration No. in SCIP database
22b5c075-11fc-41b0-ad60-dec034d8f30c <<< environmental_information.scip_declaration_number (single string!)
Country specific information

[..]

Shipping carton box

1
410 mm x 184 mm x <<< packaging_unity.length = 410, packaging_unit.width = 184

180 mm <<< packaging_unit.height = 180

[..]
"""

# %%
chat = ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0)

def convert_to_json(document_content):
    messages = [
        SystemMessage(
            content=system_message
        ),
        HumanMessage(
            content=document_content
        )
    ]

    answer = chat.invoke(messages)
    return answer.content

json = convert_to_json(document_content)

print(json)

# %%
system_message = """You analyze product descriptions to export them into a JSON format. I will present you with a product data sheet and describe the individual JSON objects and properties with <<<. You then create a JSON object from another product data sheet.

>>> Example product:

Product family benefits <<< benefits (string[])

[..]

-----

Provide your JSON in the following schema:

{
  "type": "object",
  "properties": {
    "name": {
      "type": "string"
    },
    "description": {
      "type": "string"
    },
    "applications": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "benefits": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "product_family": {
      "type": "object",
      "properties": {
        "temperature": {
          "type": "number"
        },
        "watts_min": {
          "type": "number"
        },
        "watts_max": {
          "type": "number"
        }
      }
    },
    "technical_data": {
      "type": "object",
      "properties": {
        "nominal_current": {
          "type": "number"
        },
        "control_range_min": {
          "type": "number"
        },
        "control_range_max": {
          "type": "number"
        },
        "nominal_wattage": {
          "type": "number"
        },
        "nominal_voltage": {
          "type": "number"
        }
      }
    },
    "dimensions": {
      "type": "object",
      "properties": {
        "diameter": {
          "type": "number"
        },
        "length": {
          "type": "number"
        },
        "length_base": {
          "type": "number"
        },
        "light_center_length": {
          "type": "number"
        },
        "electrode_gap": {
          "type": "number"
        },
        "weight": {
          "type": "number"
        }
      }
    },
    "operating_conditions": {
      "type": "object",
      "properties": {
        "max_temp": {
          "type": "string"
        },
        "lifespan": {
          "type": "number"
        },
        "service_lifetime": {
          "type": "number"
        }
      }
    },
    "logistical_data": {
      "type": "object",
      "properties": {
        "product_code": {
          "type": "string"
        },
        "product_name": {
          "type": "string"
        },
        "packaging_unit": {
          "type": "object",
          "properties": {
            "product_code": {
              "type": "string"
            },
            "product_name": {
              "type": "string"
            },
            "length": {
              "type": "number"
            },
            "width": {
              "type": "number"
            },
            "height": {
              "type": "number"
            },
            "volume": {
              "type": "number"
            },
            "weight": {
              "type": "number"
            }
          }
        }
      }
    }
  }
}
"""

# %%
chat = ChatOpenAI(model_name='gpt-3.5-turbo-0125', temperature=0)

def convert_to_json(document_content):
    messages = [
        SystemMessage(
            content=system_message
        ),
        HumanMessage(
            content=document_content
        )
    ]

    answer = chat.invoke(messages)
    return answer.content

json = convert_to_json(document_content)

print(json)

# %%
import os

from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# %%
pdf_files = [f for f in os.listdir('/home/sysadmin/llmops-course/AIE/Final-Project/datasets/pdf/source') if f.endswith('.pdf')]

json_documents = []

for pdf_file in pdf_files:
    with open(f'/home/sysadmin/llmops-course/AIE/Final-Project/datasets/pdf/source/{pdf_file}', 'rb') as file:
        pdf_reader = PdfReader(file)

        page_contents = []

        for page in pdf_reader.pages:
            page.extract_text(visitor_text=visitor_body)

        json = convert_to_json("\n".join(page_contents))

        json_documents.append(json)

# %%

# %%
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
docsearch = FAISS.from_texts(json_documents, embeddings)

# %%
chain = load_qa_chain(chat, chain_type="stuff", verbose=True)

query = "Can I fit the XBO 1000 W/HS OFR into a box with 350mm length and 200mm width?"

docs = docsearch.similarity_search(query)
chain.run(input_documents=docs, question=query)

# %%
import json

from typing import Any, List, Optional
from dataclasses import dataclass, field

# %%
@dataclass
class ProductFamily:
    watts_min: int
    watts_max: int
    temperature: Optional[int] = field(default=0)

    @staticmethod
    def from_dict(obj: Any) -> 'ProductFamily':
        _watts_min = int(obj.get("watts_min"))
        _watts_max = int(obj.get("watts_max"))
        _temperature = obj.get("temperature")
        return ProductFamily(_watts_min, _watts_max, _temperature)

@dataclass
class technical_data:
    nominal_current: float
    control_range_min: float
    control_range_max: float
    nominal_wattage: float
    nominal_voltage: float

    @staticmethod
    def from_dict(obj: Any) -> 'technical_data':
        _nominal_current = float(obj.get("nominal_current"))
        _control_range_min = float(obj.get("control_range_min"))
        _control_range_max = float(obj.get("control_range_max"))
        _nominal_wattage = float(obj.get("nominal_wattage"))
        _nominal_voltage = float(obj.get("nominal_voltage"))
        return technical_data(_nominal_current, _control_range_min, _control_range_max, _nominal_wattage, _nominal_voltage)
    
@dataclass
class Dimensions:
    diameter: float
    length: float
    length_base: float
    light_center_length: float
    electrode_gap: float
    weight: float

    @staticmethod
    def from_dict(obj: Any) -> 'Dimensions':
        _diameter = float(obj.get("diameter"))
        _length = float(obj.get("length"))
        _length_base = float(obj.get("length_base"))
        _light_center_length = float(obj.get("light_center_length"))
        _electrode_gap = float(obj.get("electrode_gap"))
        _weight = float(obj.get("weight"))
        return Dimensions(_diameter, _length, _length_base, _light_center_length, _electrode_gap, _weight)
    
@dataclass
class opeating_conditions:
    max_temp: str
    lifespan: float
    service_lifetime: float

    @staticmethod
    def from_dict(obj: Any) -> 'opeating_conditions':
        _max_temp = str(obj.get("max_temp"))
        _lifespan = float(obj.get("lifespan"))
        _service_lifetime = float(obj.get("service_lifetime"))
        return opeating_conditions(_max_temp, _lifespan, _service_lifetime)

@dataclass
class PackagingUnit:
    height: int
    length: int
    product_code: str
    product_name: str
    volume: float
    weight: int
    width: int

    @staticmethod
    def from_dict(obj: Any) -> 'PackagingUnit':
        assert isinstance(obj, dict)
        height = obj.get("height")
        length = obj.get("length")
        product_code = obj.get("product_code")
        product_name = obj.get("product_name")
        volume = obj.get("volume")
        weight = obj.get("weight")
        width = obj.get("width")
        return PackagingUnit(height, length, product_code, product_name, volume, weight, width)

@dataclass
class LogisticalData:
    product_code: str
    product_name: str
    packaging_unit: PackagingUnit

    @staticmethod
    def from_dict(obj: Any) -> 'LogisticalData':
        assert isinstance(obj, dict)
        product_code = obj.get("product_code")
        product_name = obj.get("product_name")
        packaging_unit = PackagingUnit.from_dict(obj.get("packaging_unit"))
        return LogisticalData(product_code, product_name, packaging_unit)

@dataclass
class Product:
    name: str
    description: str
    applications: List[str]
    benefits: List[str]
    product_family: ProductFamily
    technical_data: technical_data
    dimensions: Dimensions
    operating_conditions: opeating_conditions
    logistical_data: LogisticalData

    @staticmethod
    def from_dict(obj: Any) -> 'Product':
        _name = str(obj.get("name"))
        _description = str(obj.get("description"))
        _applicaitons = obj.get("applications")
        _benefits = obj.get("benefits")
        _product_family = ProductFamily.from_dict(obj.get("product_family"))
        _technical_data = technical_data.from_dict(obj.get("technical_data"))
        _dimensions = Dimensions.from_dict(obj.get("dimensions"))
        _opeating_conditions = opeating_conditions.from_dict(obj.get("operating_conditions"))
        _logistical_data = LogisticalData.from_dict(obj.get("logistical_data"))

        return Product(_name, _description, _applicaitons, _benefits, _product_family, _technical_data, _dimensions, _opeating_conditions, _logistical_data)


# %%
import traceback

pdf_files = [f for f in os.listdir('/home/sysadmin/llmops-course/AIE/Final-Project/datasets/pdf/source') if f.endswith('.pdf')]
products = []

for pdf_file in pdf_files:
    json_content = None
    try:
        with open(f'/home/sysadmin/llmops-course/AIE/Final-Project/datasets/pdf/source/{pdf_file}', 'rb') as file:
            pdf_reader = PdfReader(file)

            page_contents = []

            for page in pdf_reader.pages:
                page.extract_text(visitor_text=visitor_body)

            document_content = "\n".join(page_contents)

            json_content = convert_to_json(document_content)
            json_data = json.loads(json_content)

            product = Product.from_dict(json_data)
            products.append(product)
    except Exception as e:
        print("{filename} has a problem: {e}".format(filename=pdf_file, e=e))
        print(traceback.format_exc())
        print(json_content)
    else:
        os.rename(f'/home/sysadmin/llmops-course/AIE/Final-Project/datasets/pdf/source/{pdf_file}', f'/home/sysadmin/llmops-course/AIE/Final-Project/datasets/pdf/processed/{pdf_file}')

# %%
import sqlite3

# %%
if(os.path.exists('/home/sysadmin/llmops-course/AIE/Final-Project/datasets/db') == False):
    os.makedirs('/home/sysadmin/llmops-course/AIE/Final-Project/datasets/db')

db_file = '/home/sysadmin/llmops-course/AIE/Final-Project/datasets/db/products.db'

db_connection = sqlite3.connect(db_file)
db_cursor = db_connection.cursor()

db_cursor.execute('''CREATE TABLE IF NOT EXISTS Product
    (name TEXT PRIMARY KEY,
    description TEXT,
    temperature INTEGER,
    watts_min INTEGER,
    watts_max INTEGER,
    dimension_diameter REAL,
    dimension_length REAL,
    dimension_weight REAL,
    packaging_length INTEGER,
    packaging_width INTEGER,
    packaging_height INTEGER,
    packaging_weight REAL) WITHOUT ROWID
''')

db_cursor.execute('''
CREATE TABLE IF NOT EXISTS ProductApplication (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product TEXT,
    text TEXT NOT NULL,
    FOREIGN KEY (product) REFERENCES Product(name)
)
''')

db_cursor.execute('''
CREATE TABLE IF NOT EXISTS ProductBenefit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product TEXT,
    text TEXT NOT NULL,
    FOREIGN KEY (product) REFERENCES Product(name)
)
''')

db_connection.commit()

# %%
products_sql_tuples = [(
    p.name,
    p.description,
    p.product_family.temperature,
    p.product_family.watts_min,
    p.product_family.watts_max,
    p.dimensions.diameter,
    p.dimensions.length,
    p.dimensions.weight,
    p.logistical_data.packaging_unit.length,
    p.logistical_data.packaging_unit.width,
    p.logistical_data.packaging_unit.height,
    p.logistical_data.packaging_unit.weight,) for p in products]

applications_sql_tuples = []
for product in products:
    applications_sql_tuples.extend([(product.name, application) for application in product.applications])

benefits_sql_tuples = []
for product in products:
    benefits_sql_tuples.extend([(product.name, benefit) for benefit in product.benefits])

# %%
db_cursor.executemany('''
    REPLACE INTO Product (name, description, temperature, watts_min, watts_max, dimension_diameter, dimension_length, dimension_weight, packaging_length, packaging_width, packaging_height, packaging_weight)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
''', products_sql_tuples)

db_cursor.executemany('''
    REPLACE INTO ProductApplication (product, text)
    VALUES (?, ?)
''', applications_sql_tuples)

db_cursor.executemany('''
    REPLACE INTO ProductBenefit (product, text)
    VALUES (?, ?)
''', benefits_sql_tuples)

db_connection.commit()

# %%
db_cursor.close()
db_connection.close()

# %%
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

# %%
from langchain_community.agent_toolkits import create_sql_agent

db = SQLDatabase.from_uri("sqlite:///datasets/db/products.db")
llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# %%
prompt = "How many products do you have?"

result = agent_executor.invoke({"input": prompt})

# %%
prompt = "I need to find a packaging size that works for all products. What size would that package have?"

result = agent_executor.invoke({"input": prompt})

# %%
prompt = "Show me the products for brighter screen illumination"

result = agent_executor.invoke({"input": prompt})

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)

@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

    print(message.content)

    query = message.content
    
    response = agent_executor.invoke({"input": query})

    print(response['output'])
    
    msg = cl.Message(content="")

    msg.content = response['output']

    # Send and close the message stream
    await msg.send()