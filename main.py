from typing import List, Sequence
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, MessageGraph 
from chains import generate_chain, reflection_chain
from dotenv import load_dotenv
load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"message": state})

def reflection_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    result = reflection_chain.invoke({"message": state})
    return [HumanMessage(content=result.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def condition(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, condition)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()

print(graph.get_graph().draw_ascii())

if __name__ == "__main__":
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)