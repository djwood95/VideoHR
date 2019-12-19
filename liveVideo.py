import ffmpeg
import numpy as np
import png
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import colorsys
import cv2
import time
import threading
import os.path
from Util import Util

### SETTINGS ###
VIDEO_SRC = "day-70bpm-30fps.mkv" # for live video, use the corresponding webcam src (usually 0 or 2)
BUFFER_LENGTH = 3 # average together this number of frames instead of analyzing every frame
LIVE_PLOT_ENABLED = True

# how often (number of frames) to update the live plot
# if using live video, must be set to ~100, otherwise
# too many frames are dropped
LIVE_PLOT_UPDATE_FREQ = 1

cap = cv2.VideoCapture(VIDEO_SRC)
fps = cap.get(cv2.CAP_PROP_FPS)
cv2.startWindowThread()

# text settings
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (1,25)
fontScale              = .5
fontColor              = (0,0,0)
lineType               = 2

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Read the first frame to get the size
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]

util = Util()

# Determine the initial "face box"
# If box.txt does not exist, or the box is outside the size
# of the video frame, set default box to be in center of frame
try:
    f = open("box.txt", "r")
    file_contents = f.read().split('\n')
    frame_left_x, frame_right_x, frame_bottom_y, frame_top_y = file_contents[0].split()
    frame_left_x = int(frame_left_x)
    frame_right_x = int(frame_right_x)
    frame_bottom_y = int(frame_bottom_y)
    frame_top_y = int(frame_top_y)

    if frame_right_x > frame_width or frame_top_y > frame_height:
        frame_left_x, frame_right_x, frame_bottom_y, frame_top_y = util.getDefaultFaceBox(frame_width, frame_height)

    frame_width = 1

except (ValueError, IOError):
    frame_left_x, frame_right_x, frame_bottom_y, frame_top_y = util.getDefaultFaceBox(frame_width, frame_height)
    frame_width = 1

finally:
    f.close()

buffer = []
redsList = []
redsList2 = []
lightnessList = []
lightnessList2 = []

### Initialize Plots ###
fig, axis = plt.subplots(5)
plot1 = axis[0].plot(redsList)[0]
plot1a = axis[1].plot(redsList2)[0]
plot2 = axis[2].plot(lightnessList)[0]
plot3 = axis[3].plot(lightnessList2, color="red")[0]
plot4 = axis[4].plot([], [])[0]

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

## Setup Boxes ##
regBoxSetup = False
noiseBoxSetup = False
while not regBoxSetup:
    ret, frame = cap.read()
    if not ret: #reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #frameNum = 0
        ret, frame = cap.read()

    cv2.putText(frame,'Face Box Setup: Use W/A/S/D to resize, I/J/K/L to move, Space to continue', 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    c = cv2.waitKey(1)
    if c == 32: 
        regBoxSetup = True
        time.sleep(1)
        continue
    if c == 27:
        break

    if c == ord("i"): # up arrow
        frame_bottom_y -= 5
        frame_top_y -= 5
    if c == ord("j"): # left arrow
        frame_left_x -= 5
        frame_right_x -= 5
    if c == ord("k"): # down arrow
        frame_bottom_y += 5
        frame_top_y += 5
    if c == ord("l"): # right arrow
        frame_left_x += 5
        frame_right_x += 5

    if c == ord('w'): # 'grow' the box vertically
        frame_top_y -= 5
        frame_bottom_y += 5
    if c == ord('s'): # 'shrink' the box vertically
        frame_top_y += 5
        frame_bottom_y -= 5
    if c == ord('a'): # shrink horizontally
        frame_left_x += 5
        frame_right_x -= 5
    if c == ord('d'): # grow horizontally
        frame_left_x -= 5
        frame_right_x += 5

    # Draw Frame Outline
    frame[frame_bottom_y:frame_top_y, frame_left_x:frame_left_x+frame_width] = [0,255,0]
    frame[frame_bottom_y:frame_top_y+frame_width, frame_right_x:frame_right_x+frame_width] = [0,255,0]
    frame[frame_bottom_y:frame_bottom_y+frame_width, frame_left_x:frame_right_x] = [0,255,0]
    frame[frame_top_y:frame_top_y+frame_width, frame_left_x:frame_right_x] = [0,255,0]

    cv2.imshow('Face Setup', frame)

cv2.destroyAllWindows()
cv2.waitKey(1)


frameNum = 0
truthBrightness = 1
difference = 0
timestamps = []
while True:
    ret, frame = cap.read()
    frameNum += 1
    timestamps.append(time.time())
    if not ret:
        #print("Error!")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #frameNum = 0
        ret, frame = cap.read()

    # add frame to buffer, then loop back to start
    if BUFFER_LENGTH == None:
        r = np.mean(frame[frame_bottom_y:frame_top_y, frame_left_x:frame_right_x, 2])

    elif len(buffer) < (BUFFER_LENGTH-1):
        b = np.mean(frame[frame_bottom_y:frame_top_y, frame_left_x:frame_right_x, 0])
        g = np.mean(frame[frame_bottom_y:frame_top_y, frame_left_x:frame_right_x, 1])
        r = np.mean(frame[frame_bottom_y:frame_top_y, frame_left_x:frame_right_x, 2])
        buffer.append(r)
        continue
    
    else: # buffer is full, do the calculations
        r = np.mean(frame[frame_bottom_y:frame_top_y, frame_left_x:frame_right_x, 2])
        buffer.append(r)
        r = np.mean(np.asarray(buffer))
        buffer = []


    # Draw Frame Outline
    frame[frame_bottom_y:frame_top_y, frame_left_x:frame_left_x+frame_width] = [0,255,0]
    frame[frame_bottom_y:frame_top_y+frame_width, frame_right_x:frame_right_x+frame_width] = [0,255,0]
    frame[frame_bottom_y:frame_bottom_y+frame_width, frame_left_x:frame_right_x] = [0,255,0]
    frame[frame_top_y:frame_top_y+frame_width, frame_left_x:frame_right_x] = [0,255,0]

    # Add Frame to Buffer
    redsList.append( r/255.0 )

    if len(redsList) < 100:
        print("Loading frame %d/100..." % (len(redsList)))

    # if the buffer is long enough, do analysis
    if len(redsList) >= 100:

        # get the fps from the timestamp buffer
        # if the video_src is an int, then it is live
        # so we need to calculate fps here
        # otherwise use the fps from video file
        if isinstance(VIDEO_SRC, int):
            time_elapsed = timestamps[frameNum-1] - timestamps[0]
            fps = float(len(timestamps))/time_elapsed
            
        num_frames = len(redsList)

        # translate reds plot down by 0.5 so that it is centered at 0
        reds = np.array(redsList)
        reds_avg = np.mean(reds)
        reds = list(map(lambda x: x - reds_avg, reds))

        # get the frequency list
        n = np.asarray(reds).size
        freq = np.fft.fftfreq(n)

        ### apply hanning windowing function ###
        plot_reds = reds
        half = math.floor(n/2)*BUFFER_LENGTH
        x = freq[1:half]
        midpoint = np.median(x)
        x2 = np.linspace(1, num_frames, num_frames)
        norm_freq = np.hanning(num_frames)
        reds = np.multiply(reds, norm_freq)

        f = np.fft.fft(reds, norm="ortho")
        f2 = np.absolute(f)

        ### gamma correction ###
        f2 = list(map(lambda x: x**2.2, f2))

        # Pull out the best frequencies
        HR = freq[:] * fps * 60 / BUFFER_LENGTH

        # Limit HR values
        HR_filtered = []
        f2_filtered = []
        freq_filtered = []
        for i in range(len(HR)):
            if HR[i] > 30 and HR[i] < 200:
                HR_filtered.append(HR[i])
                f2_filtered.append(f2[i+1])
                freq_filtered.append(freq[i+1])

        maxIdxList = np.flip(np.argsort(f2_filtered))
        
        print("FPS = %d" % fps)
        print("Prediction #1:")
        #maxSpikeFreq = freq[ maxIdxList[0]+1 ]
        predictedHR = HR_filtered[ maxIdxList[0] ]
        maxSpikeFreq = freq_filtered[ maxIdxList[0] ] / BUFFER_LENGTH
        #predictedHR = maxSpikeFreq * fps * 60 / BUFFER_LENGTH    # XX cycles/1 frame / (1/30) = XX cycles/sec * 60 = XX cycles/min
        print("Predicted Frequency = %.3f cycles/frame" % maxSpikeFreq)
        #predictedNumCycles = maxSpikeFreq * num_frames
        #print("Predicted # Cycles in video => %.3f x %.3f = %.3f" % (maxSpikeFreq, num_frames, predictedNumCycles))
        print("Predicted HR: %.3f BPM" % predictedHR)

        cv2.putText(frame,'Predicted HR: '+str(round(predictedHR))+' bpm', 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)

        if LIVE_PLOT_ENABLED and frameNum % LIVE_PLOT_UPDATE_FREQ == 0:
            plt.clf()
            ax1 = fig.add_subplot(311)
            ax3 = fig.add_subplot(313)
            frames = list(map(lambda x: x*3, range(-num_frames, 0)))
            ax1.plot(frames, plot_reds)

            ax3.plot(HR_filtered,f2_filtered)
            plt.draw()
            plt.pause(.01)

    width = np.shape(frame)[0]
    height = np.shape(frame)[1]
    newHeight = height / (width/400)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    '''
    h = np.mean(frame_hsv[frame_bottom_y:frame_top_y, frame_left_x:frame_right_x, 0])
    s = np.mean(frame_hsv[frame_bottom_y:frame_top_y, frame_left_x:frame_right_x, 1])
    v = np.mean(frame_hsv[frame_bottom_y:frame_top_y, frame_left_x:frame_right_x, 2])

    print("h = %d" % h)
    '''

    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    #lower = np.array([h-30, s-30, v-30], dtype = "uint8")
    #upper = np.array([h+30, 255, 255], dtype = "uint8")

    # determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
    #skinMask = cv2.inRange(frame_hsv, lower, upper)

    # apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    #skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    #skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
	# mask to the frame
    #skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    #skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    c = cv2.waitKeyEx(1)

    if c == 27:
        break

    if c == 32: # space -> reset the frame buffer
        redsList = []
        lightnessList = []

    #frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
    #cv2.imshow('images', np.hstack([fullFrame, frame]))
    #cv2.resize(frame, (960, 540))
    cv2.imshow('images', frame)
    #cv2.imshow('images', frame)

f = open("box.txt", "w+")
f.write("%d %d %d %d" % (frame_left_x, frame_right_x, frame_bottom_y, frame_top_y))
f.close()

cap.release()
cv2.destroyAllWindows()