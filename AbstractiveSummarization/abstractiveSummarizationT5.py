#Import
from simplet5 import SimpleT5

#Instantiate
model = SimpleT5()

#Load trained T5 model
model.load_model("t5","AbstractiveSummarization/FineTuneModel", use_gpu=False)

#Predict
# print(model.predict())

summlist = model.predict("After a two-season stint with the Gujarat Titans, the esteemed Indian all-rounder Hardik Pandya made headlines on Monday by returning to his original IPL squad, the Mumbai Indians. Pandya had a significant presence in the MI lineup, featuring in 92 matches from 2015 to 2021. During this period, he accumulated 1476 runs at an average of 27.33 and a striking rate exceeding 153. His impactful performances included four half-centuries, with a notable top score of 91. Additionally, he contributed with the ball, clinching 42 wickets for the team, showcasing his prowess with best bowling figures of 3/20.")
print(summlist[0])