import csv

# Работает 

data = [
  ["Marker Name", "Description", "In", "Out", "Duration", "Marker Type"],
  ["", "", "00;00;14;00", "00;00;14;00", "00;00;00;00", "Comment"],
  ["hohName", "hohComment", "00;00;42;29", "00;00;42;29", "00;00;00;00", "Comment"],
  ["", "", "00;01;03;00", "00;01;03;00", "00;00;00;00", "Comment"]
]

with open('output.csv', 'w', newline='', encoding='utf-8') as file:
  writer = csv.writer(file, delimiter='\t')
  writer.writerows(data)
