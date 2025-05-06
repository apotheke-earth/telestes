from telestes.ppo import Agent

agent = Agent(input_dims=16, output_dims=2, verbose=True)
agent.save()

agent = Agent(input_dims=16, output_dims=2, load=True, verbose=True)
