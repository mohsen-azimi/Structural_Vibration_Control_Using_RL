
##########################################################################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class LoadGM:

    def __init__(self, dt, t_final, g, SF,  inputFile, outputFile, plot):
        self.resampled_dt = dt
        self.dir = 1  # for opensees
        self.t_final = t_final  # truncate ground motion to save time during training
        self.g = g
        self.SF = SF*self.g #applied during opensees run, not here
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.plot = plot

    # readPEER(self)
        dt = 0.0
        npts = 0
        self.signal = []
        inFileID = open('ground_motions\\assets\\'+self.inputFile, 'r')
        outFileID = open(self.outputFile, 'w')

        # Flag indicating dt is found and that ground motion
        # values should be read -- ASSUMES dt is on last line
        # of header!!!
        flag = 0

        # Look at each line in the file
        for line in inFileID:
            if line == '\n':
                # Blank line --> do nothing
                continue
            elif flag == 1:
                # Echo ground motion values to output file
                # outFileID.write(line)  # writes in 5 columns
                # print(type(line))
                x = line.split()
                for ii in range(0, len(x)):
                #     outFileID.write(x[ii]+'\n')  # # to write original signal in 1-col file
                    self.signal.append(x[ii])

            else:
                # Search header lines for original dt
                words = line.split()
                lengthLine = len(words)

                if lengthLine >= 4:

                    if words[0] == 'NPTS=':
                        # old SMD format
                        for word in words:
                            if word != '':
                                # Read in the time step
                                if flag == 1:
                                    dt = float(word)
                                    break

                                if flag == 2:
                                    npts = int(word.strip(','))
                                    flag = 0

                                # Find the desired token and set the flag
                                if word == 'DT=' or word == 'dt':
                                    flag = 1

                                if word == 'NPTS=':
                                    flag = 2

                    elif words[-1] == 'DT':
                        # new NGA format
                        count = 0
                        for word in words:
                            if word != '':
                                if count == 0:
                                    npts = int(word)
                                elif count == 1:
                                    dt = float(word)
                                elif word == 'DT':
                                    flag = 1
                                    break

                                count += 1


        self.original_dt = dt
        self.original_npts = npts

        self.resampled_npts = int((self.original_npts*self.original_dt)//self.resampled_dt) # n_step Analysis
        self.resampled_signal = signal.resample(self.signal, self.resampled_npts)
        self.resampled_time = np.linspace(0, self.resampled_npts*self.resampled_dt, int(self.resampled_npts*self.resampled_dt/self.resampled_dt), endpoint=True)  # analysis time-step


        # delete time-steps beyound t_final
        self.resampled_npts = int(t_final/self.resampled_dt)
        self.resampled_signal = self.resampled_signal[0:self.resampled_npts]
        self.resampled_time = self.resampled_time[0:self.resampled_npts]
        # print(len(self.resampled_signal))

        for ii in range(0, len(self.resampled_signal)):
            outFileID.write(str(self.resampled_signal[ii]) + '\n')  # writes in one column

        # print(len(self.analysis_time))
        # print(self.analysis_npts)
        if self.plot:
            plt.plot(self.resampled_time, self.resampled_signal)
            plt.xlabel('Time')
            plt.ylabel('Ground Acceleration [g]')
            plt.title(self.inputFile)
            plt.show()
            # print(len(self.signal))
            # print(len(self.resampled_signal))
            # print(self.analysis_npts)

        inFileID.close()
        outFileID.close()

# Test Class
if __name__ == "__main__":
    GM = LoadGM(dt=0.02, t_final=20, g=386., SF=1., inputFile='RSN1086_NORTHR_SYL090.AT2', outputFile='myEQ.dat', plot=True)
