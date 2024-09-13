from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts import load_prompt
# alibaba_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful Persian customer support assistant for Iran Airlines. "
#             " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
#             " When searching, be persistent.Expand your query bounds if the first search returns no results. "
#             " If a search comes up empty, expand your search before giving up. "
#             " You are going to have a conversation with two users.The first user is the MAIN USER, "
#             " who asks questions and needs to be assisted. The second user is our TOOL MANAGER, which "
#             " runs the requested tools and delivers the tool results. "
#             " You have access to the following tools to get more information if needed: "
#             " {tool_descs} "
#             " You also have access to the history of previous messages. "
#             " Generate the response in the following json format: "
#             " {{"
#             " \"THOUGHT\": \"<you should always think about what to do>\", "
#             " \"ACTION\": \"<the action to take, must be one tool_name from above tools>\", "
#             " \"ACTION_PARAMS\": \"<the input parameters to the ACTION, it must be in json format complying with the tool_params>\" "
#             " \"FINAL_ANSWER\": \"<a text containing the final answer to the original input question>\", "
#             " }} "
#             " If you don't know the answer, you can take an action using one of the provided tools. "
#             " But if you do, don't take and action and leave the action-related attributes empty. "
#             " The values `ACTION` and `FINAL_ANSWER` can never ever be filled at the same time. "
#             " If you have any questions from the user, put that in `FINAL_ANSWER` as well. "
#             " Always make sure that your output is a json complying with above format. "
#             " Do NOT add anything before or after the json response. "
#             " "
#             "\n\nCurrent user:\n\n{user_info}\n"
#             "\nCurrent time: {time}."
#         ),
#         ("placeholder", "{messages}"),
#         # HumanMessagePromptTemplate.from_template("{input}"),
#         # MessagesPlaceholder(variable_name="chat_history"),
#         # MessagesPlaceholder(variable_name="agent_scratchpad"),
#     ]
# )

# alibaba_prompt = ChatPromptTemplate.from_strings(
#     [
#         "You are a helpful Persian customer support assistant for Iran Airlines. ",
#         " You are a helpful Persian customer support assistant for Alibaba group. ",
#         " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. ",
#         " When searching, be persistent.Expand your query bounds if the first search returns no results. ",
#         " If a search comes up empty, expand your search before giving up. ",
#         " You are going to have a conversation with two users.The first user is the MAIN USER, ",
#         " who asks questions and needs to be assisted. The second user is our TOOL MANAGER, which ",
#         " runs the requested tools and delivers the tool results. ",
#         " You have access to the following tools to get more information if needed: ",
#         " {tool_descs} ",
#         " You also have access to the history of previous messages. ",
#         " Generate the response in the following json format: ",
#         " {{",
#         " \"THOUGHT\": \"<you should always think about what to do>\", ",
#         " \"ACTION\": \"<the action to take, must be one tool_name from above tools>\", ",
#         " \"ACTION_PARAMS\": \"<the input parameters to the ACTION, it must be in json format complying with the tool_params>\" ",
#         " \"FINAL_ANSWER\": \"<a text containing the final answer to the original input question>\", ",
#         " }} ",
#         " If you don't know the answer, you can take an action using one of the provided tools. ",
#         " But if you do, don't take and action and leave the action-related attributes empty. ",
#         " The values `ACTION` and `FINAL_ANSWER` can never ever be filled at the same time. ",
#         " If you have any questions from the user, put that in `FINAL_ANSWER` as well. ",
#         " Always make sure that your output is a json complying with above format. ",
#         " Do NOT add anything before or after the json response. ",
#         " ",
#         "\n\nCurrent user:\n\n{user_info}\n",
#         "\nCurrent time: {time}.",
#         " {messages}"
#         ]
# )

llama3_bartowski_instruct_prompt = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful Persian customer support assistant for Alibaba group. 
    Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
    When searching, be persistent. Expand your query bounds if the first search returns no results.
    If a search comes up empty, expand your search before giving up.
    You are going to have a conversation with two users. The first user is the MAIN USER, ho asks questions and needs to be assisted. The second user is our TOOL MANAGER, which runs the requested tools and delivers the tool results.
    You have access to the following tools to get more information if needed: 
    {tool_descs}
    You must respond ONLY with the JSON schema with the following structure:\n
    {{
        \"THOUGHT\": \"<you should always think about what to do>\",
        \"ACTION\": \"<the action to take, must be one tool_name from above tools>\",
        \"ACTION_PARAMS\": \"<the input parameters to the ACTION, it must be in json format complying with the tool_params>\",
        \"FINAL_ANSWER\": \"<a text containing the final answer to the original input question>\"
    }}\n
    Do NOT add anything before or after the json response.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {messages}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["tool_descs", "messages"],
)

llama3_bartowski_func_call_prompt = PromptTemplate(
    template="""
    <|im_start|>system You are a helpful Persian customer support assistant for Alibaba group. 
    Use the provided tools to search for flights, company policies, and other information to assist the user's queries. 
    When searching, be persistent. Expand your query bounds if the first search returns no results.
    If a search comes up empty, expand your search before giving up.
    You are going to have a conversation with two users. The first user is the MAIN USER, ho asks questions and needs to be assisted. The second user is our TOOL MANAGER, which runs the requested tools and delivers the tool results.
    You have access to the following tools to get more information if needed: 
    {tool_descs}
    You must respond ONLY with the JSON schema with the following structure:\n
    {{
        \"THOUGHT\": \"<you should always think about what to do>\",
        \"ACTION\": \"<the action to take, must be one tool_name from above tools>\",
        \"ACTION_PARAMS\": \"<the input parameters to the ACTION, it must be in json format complying with the tool_params>\",
        \"FINAL_ANSWER\": \"<a text containing the final answer to the original input question>\"
    }}\n
    Do NOT add anything before or after the json response.
    <|im_end|>
    <|im_start|>user {messages}<|im_end|>
    <|im_start|>assistant
    """,
    input_variables=["tool_descs", "user_info", "time", "messages"],
)

bartowski_llama3_8b_function_call = """
    <|im_start|>system {system_prompt}<|im_end|>
    <|im_start|>user {user_prompt}<|im_end|>
    <|im_start|>assistant
    """

bartowski_llama3_8b_instruct = " ".join([
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
    "{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>",
    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
])

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sys_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

<question>{prompt}</question><|eot_id|><|start_header_id|>assistant<|end_header_id|>

{prefill}"""

FUNC_PROMPT = """You are a helpful assistant. You are given a question inside <question> tags and a set of possible functions inside <function-definitions> tags.  
Calling these functions are optional. Carefully consider the question and determine if one or more functions can be used to answer the question. Place your thoughts and reasoning behind your decision in <function-thoughts> tags.
If the given question lacks the parameters required by the function, point it out in <function-thoughts> tags. Below is a list of function definitions:
<function-definitions>
{funcs}
</function-definitions>

If you wish to call a particular function, specify the name of the function and any arguments in a way that conforms to that function's schema inside <function-call> tags.
Function calls should be in this format: <function-thoughts>Calling func1 would be helpful because of ...</function-thoughts><function-call>[func1(params_name=params_value, params_name2=params_value2...), func2(params)]</function-call>, WITHOUT any answer.
If you do not wish to call any functions, say so in the <function-thoughts> tags followed by <function-call>None</function-call><answer>...</answer>

If and only if NO function calls are made, answer the question to the best of your ability inside <answer> tags.  If you are unsure of the answer, say so in <answer> tags.
"""

tool_desc_1 = """
    tool_name -> tavily_search_results_json
    tool_params -> query: string (search query to look up)
    tool_description -> A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.

    tool_name -> fetch_user_flight_information
    tool_params -> no parameter
    tool_description -> Fetch all tickets for the user along with corresponding flight information and seat assignments. 

    tool_name -> search_flights
    tool_params -> departure_airport: string (departure airport), arrival_airport: string (arrival airport), start_time: any of string, string (departure start time), end_time: any of string, string (departure end time), limit: integer (limit of search)
    tool_description -> Search for flights based on departure airport, arrival airport, and departure time range.

    tool_name -> search_car_rental
    tool_params -> location: string (The location of the car rental.), name: string (The name of the car rental company.), price_tier: string (The price tier of the car rental.), start_date: any of string, string (The start date of the car rental.), end_date: any of string, string (The end date of the car rental.)
    tool_description -> Search for car rentals based on location, name, price tier, start date, and end date.

    tool_name -> lookup_policy
    tool_params -> query: string (query of user to be searched in policy database.)
    tool_description -> Look up policy table of Alibaba group.

    tool_name -> search_hotels
    tool_params -> location: string (The location of the hotel.), name: string (The name of the hotel.), price_tier: string (The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury), checkin_date: any of string, string (The check-in date of the hotel.), checkout_date: any of string, string (The check-out date of the hotel.)
    tool_description -> Search for hotels based on location, name, price tier, check-in date, and check-out date..

    tool_name -> search_trip_recommendations
    tool_params -> location: string (The location of the trip recommendation.), name: string (The name of the trip recommendation.), keywords: string (The keywords associated with the trip recommendation.)
    tool_description -> Search for trip recommendations based on location, name, and keywords.

    tool_name -> update_ticket_to_new_flight
    tool_params -> ticket_no: string (ticket number), new_flight_id: integer (new flight ID)
    tool_description -> Update the user's ticket to a new valid flight

    tool_name -> cancel_ticket
    tool_params -> ticket_no: string (ticket number)
    tool_description -> Cancel the user's ticket and remove it from the database

    tool_name -> book_car_rental
    tool_params -> rental_id: integer (The ID of the car rental to book.)
    tool_description -> Search for car rentals based on location, name, price tier, start date, and end date.

    tool_name -> update_car_rental
    tool_params -> rental_id: integer (The ID of the car rental to update.), start_date: any of string, string (The new start date of the car rental.), end_date: any of string, string (The new end date of the car rental.)
    tool_description -> Update a car rental's start and end dates by its ID.

    tool_name -> cancel_car_rental
    tool_params -> rental_id: integer (The ID of the car rental to cancel.)
    tool_description -> Cancel a car rental by its ID.

    tool_name -> book_hotel
    tool_params -> hotel_id: integer (The ID of the hotel to book.)
    tool_description -> Book a hotel by its ID.

    tool_name -> update_hotel
    tool_params -> hotel_id: integer (The ID of the hotel to update.), checkin_date: any of string, string (The new check-in date of the hotel.), checkout_date: any of string, string (The new check-out date of the hotel.)
    tool_description -> Update a hotel's check-in and check-out dates by its ID.

    tool_name -> cancel_hotel
    tool_params -> hotel_id: integer (The ID of the hotel to cancel.)
    tool_description -> Cancel a hotel by its ID.

    tool_name -> book_excursion
    tool_params -> recommendation_id: integer (The ID of the trip recommendation to book.)
    tool_description -> Book a excursion by its recommendation ID.

    tool_name -> update_excursion
    tool_params -> recommendation_id: integer (The ID of the trip recommendation to update), details: string (The new details of the trip recommendation.)
    tool_description -> Update a trip recommendation's details by its ID.

    tool_name -> cancel_excursion
    tool_params -> recommendation_id: integer (The ID of the trip recommendation to cancel.)
    tool_description -> Cancel a trip recommendation by its ID.
    """

tool_desc_2 = """
{"tool_name": "tavily_search_results_json", "tool_group": "search", "tool_params": "query: string (search query to look up)", "tool_description": "A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query."}
{"tool_name": "fetch_user_flight_information", "tool_group": "flight", "tool_params": "", "tool_description": "Fetch all tickets for the user along with corresponding flight information and seat assignments. "}
{"tool_name": "search_flights", "tool_params": "tool_group": "flight", "departure_airport: string (departure airport), arrival_airport: string (arrival airport), start_time: any of string, string (departure start time), end_time: any of string, string (departure end time), limit: integer (limit of search)", "tool_description": "Search for flights based on departure airport, arrival airport, and departure time range."}
{"tool_name": "search_car_rental", "tool_group": "car", "tool_params": "location: string (The location of the car rental.), name: string (The name of the car rental company.), price_tier: string (The price tier of the car rental.), start_date: any of string, string (The start date of the car rental.), end_date: any of string, string (The end date of the car rental.)", "tool_description": "Search for car rentals based on location, name, price tier, start date, and end date."}
{"tool_name": "lookup_policy", "tool_group": "search", "tool_params": "query: string (query of user to be searched in policy database.)", "tool_description": "Look up policy table of Alibaba group."}
{"tool_name": "search_hotels", "tool_group": "hotel", "tool_params": "location: string (The location of the hotel.), name: string (The name of the hotel.), price_tier: string (The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury), checkin_date: any of string, string (The check-in date of the hotel.), checkout_date: any of string, string (The check-out date of the hotel.)", "tool_description": "Search for hotels based on location, name, price tier, check-in date, and check-out date.."}
{"tool_name": "search_trip_recommendations", "tool_group": "trip", "tool_params": "location: string (The location of the trip recommendation.), name: string (The name of the trip recommendation.), keywords: string (The keywords associated with the trip recommendation.)", "tool_description": "Search for trip recommendations based on location, name, and keywords."}
{"tool_name": "update_ticket_to_new_flight", "tool_group": "flight", "tool_params": "ticket_no: string (ticket number), new_flight_id: integer (new flight ID)", "tool_description": "Update the user's ticket to a new valid flight"}
{"tool_name": "cancel_flight_ticket", "tool_params": "tool_group": "flight", "ticket_no: string (ticket number)", "tool_description": "Cancel the user's ticket and remove it from the database"}
{"tool_name": "book_car_rental", "tool_group": "car", "tool_params": "rental_id: integer (The ID of the car rental to book.)", "tool_description": "Search for car rentals based on location, name, price tier, start date, and end date."}
{"tool_name": "update_car_rental", "tool_group": "car", "tool_params": "rental_id: integer (The ID of the car rental to update.), start_date: any of string, string (The new start date of the car rental.), end_date: any of string, string (The new end date of the car rental.)", "tool_description": "Update a car rental's start and end dates by its ID."}
{"tool_name": "cancel_car_rental", "tool_group": "car", "tool_params": "rental_id: integer (The ID of the car rental to cancel.)", "tool_description": "Cancel a car rental by its ID."}
{"tool_name": "book_hotel", "tool_group": "hotel", "tool_params": "hotel_id: integer (The ID of the hotel to book.)", "tool_description": "Book a hotel by its ID."}
{"tool_name": "update_hotel", "tool_group": "hotel", "tool_params": "hotel_id: integer (The ID of the hotel to update.), checkin_date: any of string, string (The new check-in date of the hotel.), checkout_date: any of string, string (The new check-out date of the hotel.)", "tool_description": "Update a hotel's check-in and check-out dates by its ID."}
{"tool_name": "cancel_hotel", "tool_group": "hotel", "tool_params": "hotel_id: integer (The ID of the hotel to cancel.)", "tool_description": "Cancel a hotel by its ID."}
{"tool_name": "book_excursion", "tool_group": "trip", "tool_params": "recommendation_id: integer (The ID of the trip recommendation to book.)", "tool_description": "Book a excursion by its recommendation ID."}
{"tool_name": "update_excursion", "tool_group": "trip", "tool_params": "recommendation_id: integer (The ID of the trip recommendation to update), details: string (The new details of the trip recommendation.)", "tool_description": "Update a trip recommendation's details by its ID."}
{"tool_name": "cancel_excursion", "tool_group": "trip", "tool_params": "recommendation_id: integer (The ID of the trip recommendation to cancel.)", "tool_description": " Cancel a trip recommendation by its ID."}
"""

tool_desc_opeanai = """
[
  {
    "name": "tavily_search_results_json",
    "description": "A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "description": "search query to look up",
          "type": "string"
        }
      },
      "required": [
        "query"
      ]
    }
  },
  {
    "name": "fetch_user_flight_information",
    "description": "Fetch all tickets for the user along with corresponding flight information and seat assignments. ",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  {
    "name": "search_flights",
    "description": "Search for flights based on departure airport, arrival airport, and departure time range.",
    "parameters": {
      "type": "object",
      "properties": {
        "departure_airport": {
          "description": "departure airport",
          "type": "string"
        },
        "arrival_airport": {
          "description": "arrival airport",
          "type": "string"
        },
        "start_time": {
          "description": "departure start time",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "end_time": {
          "description": "departure end time",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "limit": {
          "description": "limit of search",
          "type": "integer"
        }
      },
      "required": [
        "limit"
      ]
    }
  },
  {
    "name": "search_car_rental",
    "description": "Search for car rentals based on location, name, price tier, start date, and end date.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "description": "The location of the car rental.",
          "type": "string"
        },
        "name": {
          "description": "The name of the car rental company.",
          "type": "string"
        },
        "price_tier": {
          "description": "The price tier of the car rental.",
          "type": "string"
        },
        "start_date": {
          "description": "The start date of the car rental.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "end_date": {
          "description": "The end date of the car rental.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        }
      }
    }
  },
  {
    "name": "lookup_policy",
    "description": "Look up policy table of Alibaba group.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "description": "query of user to be searched in policy database.",
          "type": "string"
        }
      },
      "required": [
        "query"
      ]
    }
  },
  {
    "name": "search_hotels",
    "description": "Search for hotels based on location, name, price tier, check-in date, and check-out date..",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "description": "The location of the hotel.",
          "type": "string"
        },
        "name": {
          "description": "The name of the hotel.",
          "type": "string"
        },
        "price_tier": {
          "description": "The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury",
          "type": "string"
        },
        "checkin_date": {
          "description": "The check-in date of the hotel.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "checkout_date": {
          "description": "The check-out date of the hotel.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        }
      }
    }
  },
  {
    "name": "search_trip_recommendations",
    "description": "Search for trip recommendations based on location, name, and keywords.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "description": "The location of the trip recommendation.",
          "type": "string"
        },
        "name": {
          "description": "The name of the trip recommendation.",
          "type": "string"
        },
        "keywords": {
          "description": "The keywords associated with the trip recommendation.",
          "type": "string"
        }
      }
    }
  },
  {
    "name": "update_ticket_to_new_flight",
    "description": "Update the user's ticket to a new valid flight",
    "parameters": {
      "type": "object",
      "properties": {
        "ticket_no": {
          "description": "ticket number",
          "type": "string"
        },
        "new_flight_id": {
          "description": "new flight ID",
          "type": "integer"
        }
      },
      "required": [
        "ticket_no",
        "new_flight_id"
      ]
    }
  },
  {
    "name": "cancel_ticket",
    "description": "Cancel the user's ticket and remove it from the database",
    "parameters": {
      "type": "object",
      "properties": {
        "ticket_no": {
          "description": "ticket number",
          "type": "string"
        }
      },
      "required": [
        "ticket_no"
      ]
    }
  },
  {
    "name": "book_car_rental",
    "description": "Search for car rentals based on location, name, price tier, start date, and end date.",
    "parameters": {
      "type": "object",
      "properties": {
        "rental_id": {
          "description": "The ID of the car rental to book.",
          "type": "integer"
        }
      },
      "required": [
        "rental_id"
      ]
    }
  },
  {
    "name": "update_car_rental",
    "description": "Update a car rental's start and end dates by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "rental_id": {
          "description": "The ID of the car rental to update.",
          "type": "integer"
        },
        "start_date": {
          "description": "The new start date of the car rental.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "end_date": {
          "description": "The new end date of the car rental.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        }
      },
      "required": [
        "rental_id"
      ]
    }
  },
  {
    "name": "cancel_car_rental",
    "description": "Cancel a car rental by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "rental_id": {
          "description": "The ID of the car rental to cancel.",
          "type": "integer"
        }
      },
      "required": [
        "rental_id"
      ]
    }
  },
  {
    "name": "book_hotel",
    "description": "Book a hotel by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "hotel_id": {
          "description": "The ID of the hotel to book.",
          "type": "integer"
        }
      },
      "required": [
        "hotel_id"
      ]
    }
  },
  {
    "name": "update_hotel",
    "description": "Update a hotel's check-in and check-out dates by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "hotel_id": {
          "description": "The ID of the hotel to update.",
          "type": "integer"
        },
        "checkin_date": {
          "description": "The new check-in date of the hotel.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "checkout_date": {
          "description": "The new check-out date of the hotel.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        }
      },
      "required": [
        "hotel_id"
      ]
    }
  },
  {
    "name": "cancel_hotel",
    "description": "Cancel a hotel by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "hotel_id": {
          "description": "The ID of the hotel to cancel.",
          "type": "integer"
        }
      },
      "required": [
        "hotel_id"
      ]
    }
  },
  {
    "name": "book_excursion",
    "description": "Book a excursion by its recommendation ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "recommendation_id": {
          "description": "The ID of the trip recommendation to book.",
          "type": "integer"
        }
      },
      "required": [
        "recommendation_id"
      ]
    }
  },
  {
    "name": "update_excursion",
    "description": "Update a trip recommendation's details by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "recommendation_id": {
          "description": "The ID of the trip recommendation to update",
          "type": "integer"
        },
        "details": {
          "description": "The new details of the trip recommendation.",
          "type": "string"
        }
      },
      "required": [
        "recommendation_id",
        "details"
      ]
    }
  },
  {
    "name": "cancel_excursion",
    "description": " Cancel a trip recommendation by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "recommendation_id": {
          "description": "The ID of the trip recommendation to cancel.",
          "type": "integer"
        }
      },
      "required": [
        "recommendation_id"
      ]
    }
  }
]
"""

tool_desc_python = '''
def tavily_search_results_json(query: string):
    """
    A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.
    """
    Parameters:
	- query (string): search query to look up


def fetch_user_flight_information():
    """
    Fetch all tickets for the user along with corresponding flight information and seat assignments. 
    """
    Parameters:
    - user_flight_id (string): user flight id


def search_flights(departure_airport: string, arrival_airport: string, start_time: string|string, end_time: string|string, limit: integer):
    """
    Search for flights based on departure airport, arrival airport, and departure time range.
    """
    Parameters:
	- departure_airport (string): departure airport
	- arrival_airport (string): arrival airport
	- start_time (string|string): departure start time
	- end_time (string|string): departure end time
	- limit (integer): limit of search


def search_car_rental(location: string, name: string, price_tier: string, start_date: string|string, end_date: string|string):
    """
    Search for car rentals based on location, name, price tier, start date, and end date.
    """
    Parameters:
	- location (string): The location of the car rental.
	- name (string): The name of the car rental company.
	- price_tier (string): The price tier of the car rental.
	- start_date (string|string): The start date of the car rental.
	- end_date (string|string): The end date of the car rental.


def lookup_policy(query: string):
    """
    Look up policy table of Alibaba group.
    """
    Parameters:
	- query (string): query of user to be searched in policy database.


def search_hotels(location: string, name: string, price_tier: string, checkin_date: string|string, checkout_date: string|string):
    """
    Search for hotels based on location, name, price tier, check-in date, and check-out date..
    """
    Parameters:
	- location (string): The location of the hotel.
	- name (string): The name of the hotel.
	- price_tier (string): The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury
	- checkin_date (string|string): The check-in date of the hotel.
	- checkout_date (string|string): The check-out date of the hotel.


def search_trip_recommendations(location: string, name: string, keywords: string):
    """
    Search for trip recommendations based on location, name, and keywords.
    """
    Parameters:
	- location (string): The location of the trip recommendation.
	- name (string): The name of the trip recommendation.
	- keywords (string): The keywords associated with the trip recommendation.


def update_ticket_to_new_flight(ticket_no: string, new_flight_id: integer):
    """
    Update the user's ticket to a new valid flight
    """
    Parameters:
	- ticket_no (string): ticket number
	- new_flight_id (integer): new flight ID


def cancel_flight_ticket(ticket_no: string):
    """
    Cancel the user's ticket and remove it from the database
    """
    Parameters:
	- ticket_no (string): ticket number


def book_car_rental(rental_id: integer):
    """
    Search for car rentals based on location, name, price tier, start date, and end date.
    """
    Parameters:
	- rental_id (integer): The ID of the car rental to book.


def update_car_rental(rental_id: integer, start_date: string|string, end_date: string|string):
    """
    Update a car rental's start and end dates by its ID.
    """
    Parameters:
	- rental_id (integer): The ID of the car rental to update.
	- start_date (string|string): The new start date of the car rental.
	- end_date (string|string): The new end date of the car rental.


def cancel_car_rental(rental_id: integer):
    """
    Cancel a car rental by its ID.
    """
    Parameters:
	- rental_id (integer): The ID of the car rental to cancel.


def book_hotel(hotel_id: integer):
    """
    Book a hotel by its ID.
    """
    Parameters:
	- hotel_id (integer): The ID of the hotel to book.


def update_hotel(hotel_id: integer, checkin_date: string|string, checkout_date: string|string):
    """
    Update a hotel's check-in and check-out dates by its ID.
    """
    Parameters:
	- hotel_id (integer): The ID of the hotel to update.
	- checkin_date (string|string): The new check-in date of the hotel.
	- checkout_date (string|string): The new check-out date of the hotel.


def cancel_hotel(hotel_id: integer):
    """
    Cancel a hotel by its ID.
    """
    Parameters:
	- hotel_id (integer): The ID of the hotel to cancel.


def book_excursion(recommendation_id: integer):
    """
    Book a excursion by its recommendation ID.
    """
    Parameters:
	- recommendation_id (integer): The ID of the trip recommendation to book.


def update_excursion(recommendation_id: integer, details: string):
    """
    Update a trip recommendation's details by its ID.
    """
    Parameters:
	- recommendation_id (integer): The ID of the trip recommendation to update
	- details (string): The new details of the trip recommendation.


def cancel_excursion(recommendation_id: integer):
    """
     Cancel a trip recommendation by its ID.
    """
    Parameters:
	- recommendation_id (integer): The ID of the trip recommendation to cancel.
'''

tool_group_desc = """
1: Tool set: search engine, (Tool description: A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.) Tools: tavily_search_results_json 
2: Tool set: car, (Tool description: to search cars for rent, also booking and canceling car rental, and changing time slot of rental), Tools: search_car_rental, book_car_rental, update_car_rental, cancel_car_rental
3: Tool set: excursion, (Tool description: search trip, book excursion, update time slot and canceling excursion), Tools: search_trip_recommendations, book_excursion, update_excursion, cancel_excursion
4: Tool set: flight, (Tool description: to fetch ticket information, search for the available flights, update and cancel flight ticket), Tools: fetch_booked_flight_information, search_available_flights, update_ticket_to_new_flight, cancel_flight_ticket
5: Tool set: hotel, (Tool description: to search for hotels, book and cancel hotel and update check-in and check-out dates), Tools: search_hotels, book_hotel, update_hotel, cancel_hotel
6: Tool set: lookup policy, (Tool description: search in AliBaba policies), Tools: lookup_policy
"""

tool_group_desc_2 = """
1: Tool name: tavily_search_results_json, Tool description: A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query. 
2: Tool name: search_car_rental, Tool description: to search cars for rent
3: Tool name: book_car_rental, Tool description: to book a car
4: Tool name: update_car_rental, Tool description: to change time slot of a car rent
5: Tool name: cancel_car_rental, Tool description: to cancel a car rent
6: Tool name: search_trip_recommendations, Tool description: search for a trip
7: Tool name: book_excursion, Tool description: to book a excursion
8: Tool name: update_excursion, Tool description: to update a trip recommendation's details
9: Tool name: cancel_excursion, Tool description: to cancel a trip
10: Tool name: fetch_booked_flight_information, Tool description: to fetch flight ticket information
11: Tool name: search_available_flights, Tool description: to search for the available flights
12: Tool name: update_ticket_to_new_flight, Tool description: to update a flight
13: Tool name: cancel_flight_ticket, Tool description: to cancel a flight
14: Tool name: search_hotels, Tool description: to search for hotels
15: Tool name: book_hotel, Tool description: to book a hotel
16: Tool name: update_hotel, Tool description: to update check-in and check-out dates of a hotel
17: Tool name: cancel_hotel, Tool description: to cancel a hotel
18: Tool name: lookup_policy, Tool description: search in AliBaba policies
"""

# ----------------------
tool_group_prompt_2 = PromptTemplate(
    template="""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful Persian customer support assistant for Alibaba group. 
    You have access to the following tools: 
    {tool_desc}
    Base on the user query, select the tool that can be invoked to address user query, but if none of the above tools are appropriate, select "none".
    You must respond ONLY with the JSON schema with the following structure:\n
    {{
        \"TOOL\": \"<the action to take, must be one tool_name from above tools>\",
        \"THOUGHT\": \"<why did you choose this tool>\"
    }}\n
     Do NOT add anything before or after the json response.
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {messages}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["tool_descs", "messages"],
)

# ----------------------
short_tool_desc_1 = """
    tool_name -> tavily_search_results_json
    tool_params -> query: string (search query to look up)
    tool_description -> A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.

    tool_name -> fetch_user_flight_information
    tool_params -> no parameters
    tool_description -> Fetch all tickets for the user along with corresponding flight information and seat assignments. 

    tool_name -> search_flights
    tool_params -> departure_airport: string (departure airport), arrival_airport: string (arrival airport), start_time: any of string, string (departure start time), end_time: any of string, string (departure end time), limit: integer (limit of search)
    tool_description -> Search for flights based on departure airport, arrival airport, and departure time range.

    tool_name -> search_car_rental
    tool_params -> location: string (The location of the car rental.), name: string (The name of the car rental company.), price_tier: string (The price tier of the car rental.), start_date: any of string, string (The start date of the car rental.), end_date: any of string, string (The end date of the car rental.)
    tool_description -> Search for car rentals based on location, name, price tier, start date, and end date.

    tool_name -> lookup_policy
    tool_params -> query: string (query of user to be searched in policy database.)
    tool_description -> Look up policy table of Alibaba group.

    """

short_tool_desc_2 = """
[
  {
    "name": "tavily_search_results_json",
    "description": "A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "description": "search query to look up",
          "type": "string"
        }
      },
      "required": [
        "query"
      ]
    }
  },
  {
    "name": "fetch_user_flight_information",
    "description": "Fetch all tickets for the user along with corresponding flight information and seat assignments. ",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  {
    "name": "search_flights",
    "description": "Search for flights based on departure airport, arrival airport, and departure time range.",
    "parameters": {
      "type": "object",
      "properties": {
        "departure_airport": {
          "description": "departure airport",
          "type": "string"
        },
        "arrival_airport": {
          "description": "arrival airport",
          "type": "string"
        },
        "start_time": {
          "description": "departure start time",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "end_time": {
          "description": "departure end time",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "limit": {
          "description": "limit of search",
          "type": "integer"
        }
      },
      "required": [
        "limit"
      ]
    }
  }
]
"""
# --------------------

func_call_prompt_template = """
<|begin_of_text|><|start_header_id|>function_metadata<|end_header_id|>

[
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the stock price of an array of stocks",
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "An array of stocks"
                    }
                },
                "required": [
                    "names"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_big_stocks",
            "description": "Get the names of the largest N stocks by market cap",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The number of largest stocks to get the names of, e.g. 25"
                    },
                    "region": {
                        "type": "string",
                        "description": "The region to consider, can be \"US\" or \"World\"."
                    }
                },
                "required": [
                    "number"
                ]
            }
        }
    }
]<|eot_id|><|start_header_id|>user<|end_header_id|>

Get the names of the five largest stocks by market cap<|eot_id|><|start_header_id
"""

tool_params = {
  "tavily_search_results_json": {
    "name": "tavily_search_results_json",
    "description": "A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "description": "search query to look up",
          "type": "string"
        }
      },
      "required": [
        "query"
      ]
    }
  },
  "fetch_booked_flight_information": {
    "name": "fetch_user_flight_information",
    "description": "Fetch all tickets for the user along with corresponding flight information and seat assignments. ",
    "parameters": {
      "type": "object",
      "properties": {}
    }
  },
  "search_flights": {
    "name": "search_flights",
    "description": "Search for flights based on departure airport, arrival airport, and departure time range.",
    "parameters": {
      "type": "object",
      "properties": {
        "departure_airport": {
          "description": "departure airport",
          "type": "string"
        },
        "arrival_airport": {
          "description": "arrival airport",
          "type": "string"
        },
        "start_time": {
          "description": "departure start time",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "end_time": {
          "description": "departure end time",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "limit": {
          "description": "limit of search",
          "type": "integer"
        }
      },
      "required": [
        "limit"
      ]
    }
  },
  "search_car_rental": {
    "name": "search_car_rental",
    "description": "Search for car rentals based on location, name, price tier, start date, and end date.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "description": "The location of the car rental.",
          "type": "string"
        },
        "name": {
          "description": "The name of the car rental company.",
          "type": "string"
        },
        "price_tier": {
          "description": "The price tier of the car rental.",
          "type": "string"
        },
        "start_date": {
          "description": "The start date of the car rental.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "end_date": {
          "description": "The end date of the car rental.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        }
      }
    }
  },
  "lookup_policy": {
    "name": "lookup_policy",
    "description": "Look up policy table of Alibaba group.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "description": "query of user to be searched in policy database.",
          "type": "string"
        }
      },
      "required": [
        "query"
      ]
    }
  },
  "search_hotels": {
    "name": "search_hotels",
    "description": "Search for hotels based on location, name, price tier, check-in date, and check-out date..",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "description": "The location of the hotel.",
          "type": "string"
        },
        "name": {
          "description": "The name of the hotel.",
          "type": "string"
        },
        "price_tier": {
          "description": "The price tier of the hotel. Defaults to None. Examples: Midscale, Upper Midscale, Upscale, Luxury",
          "type": "string"
        },
        "checkin_date": {
          "description": "The check-in date of the hotel.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "checkout_date": {
          "description": "The check-out date of the hotel.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        }
      }
    }
  },
  "search_trip_recommendations": {
    "name": "search_trip_recommendations",
    "description": "Search for trip recommendations based on location, name, and keywords.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "description": "The location of the trip recommendation.",
          "type": "string"
        },
        "name": {
          "description": "The name of the trip recommendation.",
          "type": "string"
        },
        "keywords": {
          "description": "The keywords associated with the trip recommendation.",
          "type": "string"
        }
      }
    }
  },
  "update_ticket_to_new_flight": {
    "name": "update_ticket_to_new_flight",
    "description": "Update the user's ticket to a new valid flight",
    "parameters": {
      "type": "object",
      "properties": {
        "ticket_no": {
          "description": "ticket number",
          "type": "string"
        },
        "new_flight_id": {
          "description": "new flight ID",
          "type": "integer"
        }
      },
      "required": [
        "ticket_no",
        "new_flight_id"
      ]
    }
  },
  "cancel_flight_ticket": {
    "name": "cancel_flight_ticket",
    "description": "Cancel the user's ticket and remove it from the database",
    "parameters": {
      "type": "object",
      "properties": {
        "ticket_no": {
          "description": "ticket number",
          "type": "string"
        }
      },
      "required": [
        "ticket_no"
      ]
    }
  },
  "book_car_rental": {
    "name": "book_car_rental",
    "description": "Search for car rentals based on location, name, price tier, start date, and end date.",
    "parameters": {
      "type": "object",
      "properties": {
        "rental_id": {
          "description": "The ID of the car rental to book.",
          "type": "integer"
        }
      },
      "required": [
        "rental_id"
      ]
    }
  },
  "update_car_rental": {
    "name": "update_car_rental",
    "description": "Update a car rental's start and end dates by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "rental_id": {
          "description": "The ID of the car rental to update.",
          "type": "integer"
        },
        "start_date": {
          "description": "The new start date of the car rental.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "end_date": {
          "description": "The new end date of the car rental.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        }
      },
      "required": [
        "rental_id"
      ]
    }
  },
  "cancel_car_rental": {
    "name": "cancel_car_rental",
    "description": "Cancel a car rental by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "rental_id": {
          "description": "The ID of the car rental to cancel.",
          "type": "integer"
        }
      },
      "required": [
        "rental_id"
      ]
    }
  },
  "book_hotel": {
    "name": "book_hotel",
    "description": "Book a hotel by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "hotel_id": {
          "description": "The ID of the hotel to book.",
          "type": "integer"
        }
      },
      "required": [
        "hotel_id"
      ]
    }
  },
  "update_hotel": {
    "name": "update_hotel",
    "description": "Update a hotel's check-in and check-out dates by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "hotel_id": {
          "description": "The ID of the hotel to update.",
          "type": "integer"
        },
        "checkin_date": {
          "description": "The new check-in date of the hotel.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        },
        "checkout_date": {
          "description": "The new check-out date of the hotel.",
          "anyOf": [
            {
              "type": "string",
              "format": "date"
            },
            {
              "type": "string",
              "format": "date-time"
            }
          ]
        }
      },
      "required": [
        "hotel_id"
      ]
    }
  },
  "cancel_hotel": {
    "name": "cancel_hotel",
    "description": "Cancel a hotel by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "hotel_id": {
          "description": "The ID of the hotel to cancel.",
          "type": "integer"
        }
      },
      "required": [
        "hotel_id"
      ]
    }
  },
  "book_excursion": {
    "name": "book_excursion",
    "description": "Book a excursion by its recommendation ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "recommendation_id": {
          "description": "The ID of the trip recommendation to book.",
          "type": "integer"
        }
      },
      "required": [
        "recommendation_id"
      ]
    }
  },
  "update_excursion": {
    "name": "update_excursion",
    "description": "Update a trip recommendation's details by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "recommendation_id": {
          "description": "The ID of the trip recommendation to update",
          "type": "integer"
        },
        "details": {
          "description": "The new details of the trip recommendation.",
          "type": "string"
        }
      },
      "required": [
        "recommendation_id",
        "details"
      ]
    }
  },
  "cancel_excursion": {
    "name": "cancel_excursion",
    "description": " Cancel a trip recommendation by its ID.",
    "parameters": {
      "type": "object",
      "properties": {
        "recommendation_id": {
          "description": "The ID of the trip recommendation to cancel.",
          "type": "integer"
        }
      },
      "required": [
        "recommendation_id"
      ]
    }
  }
}

prompt_template = """
<|begin_of_text|><|start_header_id|>function_metadata<|end_header_id|>
{tool_desc}
<|eot_id|><|start_header_id|>system<|end_header_id|>

<|eot_id|><|start_header_id|>user<|end_header_id|>
{messages}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

if __name__ == "__main__":
  # prompt = load_prompt('lc://prompts/path/to/file.json')
  from langchain import hub
