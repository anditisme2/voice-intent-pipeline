# CodeBud
An intelligent coding companion that generates code from natural language prompts and then explains the generated code to the user in an easy-to-understand language. 
Additionally, it can also assist in debugging and code correction. 
## Modules
**Module 1:** Natural Language Understanding (NLU) Pipeline  
***BiLSTM-CRF architecture trained on the SNIPS NLU dataset***  
The NLU module is the entry point of CodeBud. Before any code can be generated, the system must first understand what the user is asking — this is Module 1's job.
Given a prompt like:
> "Write a Python function to find factorial"
The module extracts two things:  
* Intent — what the user wants to do (e.g. generate_code, debug_code, explain_code)  
* Slots — the parameters of that intent (e.g. language: Python, task: factorial function)  
This structured output is then passed downstream to the code generation engine.
