from agentchord.gbc_object import GBC

str_example_1 = GBC("Hello, this is a GBCStr object!", connections=["connection1", "connection2"], weights=[0.5, 0.5], subject="ExampleAgent")
print(str_example_1.__class__)
print(str_example_1.get_node_representation())
print(str_example_1.get_connections())

str_example_2 = GBC("This is another GBCStr object.", connections=["connection3"], weights=[1.0], subject="AnotherExampleAgent")
print(str_example_2.__class__)
print(str_example_2.get_node_representation())
print(str_example_2.get_connections())

str_example_3 = str_example_1 + str_example_2
print(str_example_3.__class__)
print(str_example_3.get_node_representation())
print(str_example_3.get_connections())

str_example_4 = str_example_1[:10]
print(str_example_4.__class__)
print(str_example_4.get_node_representation())
print(str_example_4.get_connections())