#!/usr/bin/env python
# coding: utf-8

# In[2]:


import subprocess


# In[3]:


def main():

    print("Starter full bysykkel-pipeline...\n")

    print("Trener modell...")
    subprocess.run(["python", "train.py"], check=True)
    print("Modell ferdig trent og lagret.\n")

    print("Kjører prediksjon for neste time...")
    subprocess.run(["python", "predict.py"], check=True)
    print("\nPrediksjon ferdig!\n")

    print("Hele pipelinen er fullført!")

if __name__ == "__main__":
    main()


# In[ ]:




