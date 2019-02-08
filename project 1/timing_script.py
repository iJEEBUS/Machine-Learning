import time
import MLToolkit
import os

iterations = 100



def time_script():
    script_avg_time = 0.0
    script_times = []


    print("\nExecuting script version....")

    for i in range(0, iterations):
      start_time = time.process_time()
      os.system(" python3 ./DataAnalysis.py")
      script_times.append(time.process_time() - start_time)

    script_avg_time = sum(script_times)/iterations
    print("\nAverage time: ", script_avg_time)
    print("Max: ", max(script_times))
    print("Min ", min(script_times), '\n')


def time_toolkit():
    print("\nExecuting toolkit version...")
    toolkit_avg_time = 0
    toolkit_times = []

    def toolkit():
      tools = MLToolkit.MLToolkit()
      tools.load_data("./data.txt", filter="delete")
      num_obs, intercept, slope, LOBF_slr = tools.SLR()

    for i in range(0, iterations):
      start_time = time.process_time()
      toolkit()
      toolkit_times.append(time.process_time() - start_time)

    toolkit_avg_time = sum(toolkit_times)/iterations
    print("\nAverage time: ", toolkit_avg_time)
    print("Max: ", max(toolkit_times))
    print("Min ", min(toolkit_times), '\n')




time_script()
time_toolkit()
