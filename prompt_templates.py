intent_prompt = """You are an expert assistant helping detect user intents for a food pantry search system in Kansas.

Given a user's message, identify one or more relevant intents from the following list:

[
  "Find Pantry by County",
  "Find Pantry by City or Town",
  "Find Pantry by Open Hours",
  "Find Student-Only Pantry",
  "Find Pantry with No ID Required",
  "Find TEFAP Site Pantry",
  "Find Pantry Requiring Proof of Residency",
  "Find Mobile Pantry",
  "Find Pantry Contact Information",
  "Find Pantry for Seniors",
  "Find Pantry by Appointment Requirement"
]

Return your answer as a JSON array of the matched intents.

If no intents are matched, return an empty array [].

Examples:

User: "Where can I find a pantry open tomorrow in Wichita?"
Answer: ["Find Pantry by City or Town", "Find Pantry by Open Hours"]

User: "Give me a mobile pantry in Allen County."
Answer: ["Find Pantry by County", "Find Mobile Pantry"]

User: "Is there a pantry that doesn't ask for ID?"
Answer: ["Find Pantry with No ID Required"]

Now, detect intents for this message:

User: "{query}"
Answer:"""
