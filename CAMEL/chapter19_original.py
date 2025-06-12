#LLM

#CAMEL:  Can LLM generate these guiding texts on its own?
"""
CAMEL Interaction Agent Framework
CAMEL: Communicative Agents for “Mind” Exploration of Large Language Model Society

CAMEL:Comminication Agents Mind Exploration LLM

Comminication Agents:

Inception Prompting: Task specifier prompt\ AI assistant prompts and AI user prompts

Task: Brainstorming a flower maeketing scheme through role-playing.-
1. Role-playing
2. Task Specification
3. Initial prompt setup
4. Interaction norms
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

        output_message = self.model(messages)
        self.update_messages(output_message)

        return output_message
#2. Preset roles and task prompts
assistant_role_name = "Flower Shop Marketing Specialist"
user_role_name = "Flower Shop Owner"
task = "Develop a marketing strategy for a summer rose night event."
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
        f"{user_sys_msg.content}。"
        "Now start introducing them to me one by one."
        "Only reply with instructions and inputs."
    )
)

user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
user_msg = assistant_agent.step(user_msg)

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


#hw: Are there any business scenarios that need to be refined and specified in your requirements?Please apply the CAMEL code implement them.

"""
AI User (Flower Shop Owner):

Instruction: Provide an overview of the targeted demographics for the Summer Rose Night event.
Input: The event aims to attract adults aged 25-45 who appreciate floral arrangements, romantic settings, and social events. The target audience is predominantly female, with an interest in upscale experiences and social gatherings.


AI Assistant (Flower Shop Marketing Specialist):

Solution: The targeted demographics for the Summer Rose Night event are adults aged 25-45, predominantly female, who appreciate floral arrangements, romantic settings, upscale experiences, and social gatherings. 

Next request.


AI User (Flower Shop Owner):

Instruction: Outline a social media campaign strategy for the Summer Rose Night event.
Input: The social media campaign should focus on platforms like Instagram and Facebook to showcase the beauty of roses, promote the event's romantic ambiance, and highlight any special offers or activities. Visual content such as photos and videos of the venue, floral arrangements, and past events should be included. Engagement tactics like contests, polls, and behind-the-scenes sneak peeks can help generate buzz.


AI Assistant (Flower Shop Marketing Specialist):

Solution: Develop a social media campaign strategy for the Summer Rose Night event by focusing on platforms like Instagram and Facebook. Showcase the beauty of roses, promote the event's romantic ambiance, and highlight special offers or activities through visual content such as venue photos, floral arrangements, and past event videos. Implement engagement tactics like contests, polls, and behind-the-scenes sneak peeks to generate buzz and increase event awareness.

 Next request.


AI User (Flower Shop Owner):

Instruction: Propose ideas for print advertisements to promote the Summer Rose Night event.
Input: Print advertisements could include flyers distributed in local cafes, boutiques, and beauty salons, as well as posters displayed in high-traffic areas. The design should feature elegant rose imagery, event details, and a call-to-action to purchase tickets or RSVP. Consider partnering with local magazines or newspapers for ad placements to reach a broader audience.


AI Assistant (Flower Shop Marketing Specialist):

Solution: Create print advertisements for the Summer Rose Night event by distributing flyers in local cafes, boutiques, and beauty salons, and displaying posters in high-traffic areas. Design the ads with elegant rose imagery, event details, and a clear call-to-action to purchase tickets or RSVP. Explore partnership opportunities with local magazines or newspapers for additional ad placements to reach a wider audience. Next request.


AI User (Flower Shop Owner):

Instruction: Suggest potential promotional partnerships for the Summer Rose Night event.
Input: Consider partnering with local florists for floral arrangements, wineries for wine tasting, restaurants for catering, and event planners for coordination. Collaborating with beauty salons for special offers, photographers for event coverage, and influencers for promotion can also enhance 
the event experience and attract a diverse audience.


AI Assistant (Flower Shop Marketing Specialist):

Solution: Explore potential promotional partnerships for the Summer Rose Night event by collaborating with local florists for floral arrangements, wineries for wine tasting, restaurants for catering, and event planners for coordination. Additionally, consider partnering with beauty salons for special offers, photographers for event coverage, and influencers for promotion to enhance the event experience and attract a diverse audience. Next request.


AI User (Flower Shop Owner):

Instruction: Create a timeline for the implementation of the marketing strategy for the Summer Rose Night event.
Input:
- 3 months before the event: Launch social media campaign, finalize print advertisements, and secure promotional partnerships.
- 2 months before the event: Distribute print advertisements, continue social media promotions, and confirm event logistics.
- 1 month before the event: Intensify social media engagement, coordinate with partners for final preparations, and send out event reminders.
- 1 week before the event: Ramp up social media buzz, conduct final venue walkthroughs, and ensure all materials are ready.
- Day of the event: Execute marketing plan, monitor social media for engagement, and oversee event operations.
- Post-event: Share highlights on social media, thank partners for their collaboration, and gather feedback for future improvements.

AI Assistant (Flower Shop Marketing Specialist):

Solution: Develop a timeline for the marketing strategy of the Summer Rose Night event as follows:
- 3 months before the event: Launch social media campaign, finalize print advertisements, and secure promotional partnerships.
- 2 months before the event: Distribute print advertisements, continue social media promotions, and confirm event logistics.
- 1 month before the event: Intensify social media engagement, coordinate with partners for final preparations, and send out event reminders.
- 1 week before the event: Ramp up social media buzz, conduct final venue walkthroughs, and ensure all materials are ready.
- Day of the event: Execute marketing plan, monitor social media for engagement, and oversee event operations.
- Post-event: Share highlights on social media, thank partners for their collaboration, and gather feedback for future improvements. Next request.


AI User (Flower Shop Owner):

<CAMEL_TASK_DONE>


AI Assistant (Flower Shop Marketing Specialist):

Task complete.

"""