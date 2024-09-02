# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:10:10 2024

@author: Lenovo
"""

import os

# Directory and file structure for RASA chatbot project
# os.makedirs('dentist_bot', exist_ok=True)
# os.chdir('dentist_bot')

# Writing configuration files
config_yml = """
language: en
pipeline:
  - name: WhitespaceTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: DIETClassifier
    epochs: 100
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
  - name: FallbackClassifier
    threshold: 0.3
    ambiguity_threshold: 0.1
policies:
  - name: MemoizationPolicy
  - name: TEDPolicy
    max_history: 5
    epochs: 100
  - name: RulePolicy
"""

domain_yml = """
intents:
  - greet
  - ask_address
  - ask_timings
  - book_appointment
  - affirm
  - deny

entities:
  - time
  - date

slots:
  time:
    type: text
    mappings:
      - type: from_entity
        entity: time
  date:
    type: text
    mappings:
      - type: from_entity
        entity: date

responses:
  utter_greet:
    - text: "Hello! How can I help you today?"
  utter_address:
    - text: "Our clinic is located at 123 Dental Street."
  utter_timings:
    - text: "We are open from 9 AM to 5 PM, Monday to Saturday."
  utter_ask_date:
    - text: "When would you like to book the appointment?"
  utter_ask_time:
    - text: "At what time would you prefer?"
  utter_confirm_appointment:
    - text: "Your appointment is booked for {date} at {time}."
  utter_fallback:
    - text: "I'm sorry, I didn't understand that. Can you please rephrase?"

actions:
  - action_book_appointment

session_config:
  session_expiration_time: 60
  carry_over_slots_to_new_session: True
"""

data_yml = """
version: "2.0"
nlu:
  - intent: greet
    examples: |
      - Hi
      - Hello
      - Hey there
  - intent: ask_address
    examples: |
      - Where is the clinic located?
      - What is the address?
  - intent: ask_timings
    examples: |
      - What are your working hours?
      - When are you open?
  - intent: book_appointment
    examples: |
      - I want to book an appointment
      - Can I schedule a visit?
  - intent: affirm
    examples: |
      - Yes
      - Sure
  - intent: deny
    examples: |
      - No
      - Not now

stories:
  - story: book an appointment
    steps:
      - intent: book_appointment
      - action: utter_ask_date
      - intent: inform
        entities:
          - date: "2023-12-01"
      - action: utter_ask_time
      - intent: inform
        entities:
          - time: "10:00 AM"
      - action: action_book_appointment
      - action: utter_confirm_appointment

rules:
  - rule: greet the user
    steps:
      - intent: greet
      - action: utter_greet
  - rule: provide address
    steps:
      - intent: ask_address
      - action: utter_address
  - rule: provide timings
    steps:
      - intent: ask_timings
      - action: utter_timings
  - rule: fallback
    steps:
      - intent: nlu_fallback
      - action: utter_fallback
"""

action_py = """
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionBookAppointment(Action):
    def name(self) -> Text:
        return "action_book_appointment"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        date = tracker.get_slot('date')
        time = tracker.get_slot('time')
        dispatcher.utter_message(text=f"Your appointment is booked for {date} at {time}.")
        return []
"""

# Writing files to the project directory
with open('config.yml', 'w') as file:
    file.write(config_yml)

with open('domain.yml', 'w') as file:
    file.write(domain_yml)

with open('data/nlu.yml', 'w') as file:
    file.write(data_yml)

os.makedirs('actions', exist_ok=True)
with open('actions/actions.py', 'w') as file:
    file.write(action_py)

print("RASA chatbot project structure created successfully.")
