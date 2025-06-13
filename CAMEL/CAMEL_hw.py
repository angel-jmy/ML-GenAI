#LLM
#CAMEL: Can ChatGPT generate these guiding texts on its own?
"""
CAMEL: Interaction Agent Framework
CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society
CAMEL: Communication Agents Mind Exploration LLM

Communication Agents:

Inception Prompting: Task specifier prompt / AI assistant prompts and AI user prompts

Task: Brainstorming a flower marketing scheme through role-playing.
1. Role-playing Agent
2. Task Specification
3. Initial prompt setup
4. Interaction norms

arxiv.org/pdf/2303.17760

"""

from dotenv import load_dotenv
load_dotenv()

from typing import List
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)

#1. Define the CAMEL Agent class
class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(self, input_message: HumanMessage) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model.invoke(messages)
        self.update_messages(output_message)

        return output_message
#2. Preset roles and task prompts
assistant_role_name = "Transportation Researcher at an institute of transportation studies"
user_role_name = "Transportation Policy Manager at a State DOT or Air Resources Board"
task = "Assess the feasibility and equity implications of implementing congestion pricing in the downtown area of San Francisco."
word_limit = 50

#Tasks assignment to agents-Task Specifier
task_specifier_sys_msg = SystemMessage(content="You can make the task more specific.")
task_specifier_prompt = """This is a task that {assistant_role_name} will help {user_role_name} complete: {task}.
Please make it more specific. Use your creativity and imagination.
Reply with a specific task in {word_limit} words or less. Do not add anything else."""

task_specifier_template = HumanMessagePromptTemplate.from_template(
    template=task_specifier_prompt
)
task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(model_name = 'gpt-4', temperature=1.0))
task_specifier_msg = task_specifier_template.format_messages(
    assistant_role_name=assistant_role_name,
    user_role_name=user_role_name,
    task=task,
    word_limit=word_limit,
)[0]
specified_task_msg = task_specify_agent.step(task_specifier_msg)
print(f"Specified task: {specified_task_msg.content}")
specified_task = specified_task_msg.content
"""
Specified task: Identify and analyze the target demographic for the summer rose night event, plan a social media promotion strategy including ads, posts, and partnerships, and prepare attractive promotional materials like banners, flyers, and online visuals.
Original task prompt:Develop a marketing strategy for a summer rose night event.
"""

#System message tamplates
assistant_inception_prompt = """Never forget that you are {assistant_role_name}, and I am {user_role_name}. Never reverse roles! Never instruct me!
We have a common interest, which is to successfully complete the task together.
You must help me complete the task.
This is the task: {task}. Never forget our task!
I must instruct you to complete the task based on your expertise and my needs.

I can only give you one instruction at a time.
You must write a specific solution to the requested instruction.
If you are unable to execute the instruction due to physical, moral, legal reasons, or your abilities, you must honestly refuse my instruction and explain the reasons.
Do not add anything other than the solution to my instruction.
You should never ask me any questions; you only answer questions.
You should never respond with an ambiguous solution. Explain your solution.
Your solution must be a statement and use simple present tense.
You should always start with the following unless I say the task is complete:

Solution: <YOUR_SOLUTION>

<YOUR_SOLUTION> should be specific and provide the preferred implementation and examples to solve the task.
Always end <YOUR_SOLUTION> with "Next request".
"""

user_inception_prompt = """Never forget that you are {user_role_name}, and I am {assistant_role_name}. Never switch roles! You always guide me.
Our common goal is to successfully complete a task together.
I must help you complete this task.
This is the task: {task}. Never forget our task!
You can guide me in two ways based on my expertise and your needs:

1. Provide necessary input to guide:
Instruction: <YOUR_INSTRUCTION>
Input: <YOUR_INPUT>

2. Provide no input to guide:
Instruction: <YOUR_INSTRUCTION>
Input: None

The "Instruction" describes a task or question. The paired "Input" provides further background or information for the requested "Instruction".

You must give me one instruction at a time.
I must write a reply appropriately completing the requested instruction.
If I am unable to execute your instruction due to physical, moral, legal reasons, or my abilities, I must honestly refuse your instruction and explain the reasons.
You should guide me instead of asking me questions.
Now you must start guiding me in the two ways described above.
Do not add anything other than your instructions and optional corresponding inputs!
Continue giving me instructions and necessary inputs until you believe the task is complete.
When the task is complete, simply reply with a single word <CAMEL_TASK_DONE>.
Never say <CAMEL_TASK_DONE> unless my response has addressed your task.
"""


def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=assistant_inception_prompt
    )
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=user_inception_prompt
    )
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    return assistant_sys_msg, user_sys_msg

assistant_sys_msg, user_sys_msg = get_sys_msgs(
    assistant_role_name, user_role_name, specified_task
)

#Create Agent instance
assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

assistant_agent.reset()
user_agent.reset()

assistant_msg = HumanMessage(
    content=(
        f"{user_sys_msg.content}."
        "Now start introducing them to me one by one."
        "Only reply with instructions and inputs."
    )
)

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
# user_msg = assistant_agent.step(user_msg)

print(f"Original task prompt:\n{task}\n")
print(f"Specified task prompt:\n{specified_task}\n")
"""
Original task prompt:
Develop a marketing strategy for a summer rose night event.

Specified task prompt:
Develop a marketing strategy for a Summer Rose Night event, outlining social media campaigns, local press outreach, in-store promotion tactics, and customer engagement ideas, specifically aimed at boosting event attendance and rose sales.

"""

#begin brainstorming
chat_turn_limit, n = 30, 0
while n < chat_turn_limit:
    n += 1
    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")

    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    if "<CAMEL_TASK_DONE>" in user_msg.content:
        break





# '''
# Summarizing the action items
# '''

summary_pairs = []
messages = assistant_agent.stored_messages

# Start from index 1 to skip the system message
for i in range(1, len(messages), 2):
    if isinstance(messages[i], HumanMessage) and i + 1 < len(messages):
        instruction = messages[i].content
        solution = messages[i + 1].content
        summary_pairs.append(f"Instruction: {instruction}\nSolution: {solution}")


summary_prompt = (
    "You are a helpful assistant. Summarize the following instruction-solution pairs "
    "into a list of clear action items for a project implementation checklist.\n\n"
    "Respond with a bullet point list of specific, unambiguous actions.\n\n"
    "Each bullet should not exceed 20 words in length.\n\n"
    "Conversation History:\n\n"
    + "\n\n---\n\n".join(summary_pairs)
)



summary_llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

response = summary_llm.invoke([
    HumanMessage(content=summary_prompt)
])

print("✅ Final Action Items:\n")
print(response.content)



#hw: Are there any business scenarios that need to be refined and specified in your requirements? Please apply the CAMEL code implement them.


'''
✅ Final Action Items:

- Analyze current traffic congestion levels, peak hours, major bottlenecks, and existing congestion management strategies in San Francisco's CBD.
- Examine demographic data to identify population distribution and trends in San Francisco's CBD and surrounding areas.
- Analyze commuting patterns, including modes of transportation used, origin-destination pairs, peak commuting hours, and commuting trends in San Francisco's CBD.
- Analyze vehicle emissions data, pollutant levels, sources of emissions, and their impact on air quality in San Francisco's CBD.
- Assess potential changes in travel behavior, economic implications for businesses, and effects on low-income residents due to congestion pricing in San Francisco's CBD.
- Devise mitigation strategies to address potential equity implications for low-income residents resulting from congestion pricing.
- Evaluate the cost-effectiveness and feasibility of implementing congestion pricing in San Francisco's CBD.
- Summarize the findings and recommendations based on the evaluation of the socioeconomic impact, cost-effectiveness, and feasibility of implementing congestion pricing.
- Plan for further analysis of detailed implementation plans, continued stakeholder engagement, and pilot testing for successful deployment of congestion pricing.
'''