from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.save_context({"input": "Hi"}, {"output": "How are you?"})

rst = memory.load_memory_variables({})
print(rst)


from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    return_messages=True,
    k=4
)

def add_message(input, output):
    memory.save_context({"input": input}, {"output": output})

add_message(1, 1)
add_message(2, 2)
add_message(3, 3)
add_message(4, 4)

rst = memory.load_memory_variables({})
print(rst)

add_message(5, 5)
rst = memory.load_memory_variables({})
print(rst)

