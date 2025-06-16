from agentchord.gbc_object import GBC, visualize_gbc_tree

list_example_1 = GBC(
    [1, 2, 3, 4, 5],
    connections=["connection1", "connection2"],
    weights=[0.5, 0.5],
    subject="ExampleAgent"
)
print(list_example_1.__class__)
print(list_example_1.get_node_representation())
print(list_example_1.get_connections())

list_example_2 = GBC(
    [6, 7, 8],
    connections=["connection3"],
    weights=[1.0],
    subject="AnotherExampleAgent"
)
print(list_example_2.__class__)
print(list_example_2.get_node_representation()) 
print(list_example_2.get_connections())

list_example_3 = list_example_1 + list_example_2
print(list_example_3.__class__)
print(list_example_3.get_node_representation())
print(list_example_3.get_connections())

list_example_4 = list_example_1[:3]
print(list_example_4.__class__)
print(list_example_4.get_node_representation())
print(list_example_4.get_connections())

list_example_4.append(9)
print(list_example_4.__class__)
print(list_example_4.get_node_representation())
print(list_example_4.get_connections())

list_example_4.extend([10, 11])
print(list_example_4.__class__)
print(list_example_4.get_node_representation())
print(list_example_4.get_connections())

visualize_gbc_tree(list_example_4)