import pandas as pd

data = pd.read_csv('grasp-rate-prediction-step.txt')


result = "<tbody>\n"

for i, row in data.iterrows():
    tr = f"""  <tr>
    <td>{i}</td>
    <td>{row.step}</td>
    <td style="text-align: right">{row.reward:0.3f}</td>
    <td style="text-align: right">{row.estimated_reward:0.3f}</td>
    <td style="text-align: right">{row.estimated_reward_std:0.3f}</td>
  </tr>
"""
    result += tr


result += "</tbody>"

with open('conv.txt', 'w') as file:
    file.write(result)
