#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on mars 21, 2025, at 16:03
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code


# Run 'Before Experiment' code from code_3


# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'TD_motor_variability'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'group': '1',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1536, 960]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\doria\\Desktop\\test_td\\TD_motor_variability.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Onset_screen" ---
    onset_rect = visual.Rect(
        win=win, name='onset_rect',
        width=(0.25, 0.1)[0], height=(0.25, 0.1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    mouse_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_3.mouseClock = core.Clock()
    text = visual.TextStim(win=win, name='text',
        text='Click here to start',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "buffer" ---
    polygon_2 = visual.ShapeStim(
        win=win, name='polygon_2',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "fixation_cross" ---
    polygon_7 = visual.ShapeStim(
        win=win, name='polygon_7', vertices='cross',
        size=(0.045, 0.045),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    target1_4 = visual.ShapeStim(
        win=win, name='target1_4',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(-0.8, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    target2_4 = visual.ShapeStim(
        win=win, name='target2_4',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(-0.8, 0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    target3_4 = visual.ShapeStim(
        win=win, name='target3_4',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(0.8, 0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-3.0, interpolate=True)
    target4_4 = visual.ShapeStim(
        win=win, name='target4_4',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(0.8, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_error_feedback" ---
    # Run 'Begin Experiment' code from code
    
    block1_seq = [0,0,1,1,2,2,3,3]
    shuffle(block1_seq)
    print(block1_seq)
    block1_count = 0
    print("block count", block1_count)
    
    
    target1 = visual.ShapeStim(
        win=win, name='target1',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(-0.8, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    target2 = visual.ShapeStim(
        win=win, name='target2',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(-0.8, 0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    target3 = visual.ShapeStim(
        win=win, name='target3',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(0.8, 0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    target4 = visual.ShapeStim(
        win=win, name='target4',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(0.8, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "interblockscreen" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Interblock screen\nClick anywhere to move one',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse_6 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_6.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "Onset_screen" ---
    onset_rect = visual.Rect(
        win=win, name='onset_rect',
        width=(0.25, 0.1)[0], height=(0.25, 0.1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    mouse_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_3.mouseClock = core.Clock()
    text = visual.TextStim(win=win, name='text',
        text='Click here to start',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "buffer" ---
    polygon_2 = visual.ShapeStim(
        win=win, name='polygon_2',
        size=(0.5, 0.5), vertices='triangle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "fixation_cross" ---
    polygon_7 = visual.ShapeStim(
        win=win, name='polygon_7', vertices='cross',
        size=(0.045, 0.045),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    target1_4 = visual.ShapeStim(
        win=win, name='target1_4',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(-0.8, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-1.0, interpolate=True)
    target2_4 = visual.ShapeStim(
        win=win, name='target2_4',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(-0.8, 0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    target3_4 = visual.ShapeStim(
        win=win, name='target3_4',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(0.8, 0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-3.0, interpolate=True)
    target4_4 = visual.ShapeStim(
        win=win, name='target4_4',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(0.8, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_reward_feedback" ---
    # Run 'Begin Experiment' code from code_3
    block2_seq = [0,0,1,1,2,2,3,3]
    shuffle(block2_seq)
    print(block2_seq)
    block2_count = 0
    print("block count", block2_count)
    target1_2 = visual.ShapeStim(
        win=win, name='target1_2',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(-0.8, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-2.0, interpolate=True)
    target2_2 = visual.ShapeStim(
        win=win, name='target2_2',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(-0.8, 0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-3.0, interpolate=True)
    target3_2 = visual.ShapeStim(
        win=win, name='target3_2',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(0.8, 0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-4.0, interpolate=True)
    target4_2 = visual.ShapeStim(
        win=win, name='target4_2',units='norm', 
        size=(0.15, 0.2), vertices='circle',
        ori=0.0, pos=(0.8, -0.8), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-5.0, interpolate=True)
    mouse_4 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_4.mouseClock = core.Clock()
    polygon_4 = visual.ShapeStim(
        win=win, name='polygon_4',
        size=(0.7, 0.7), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[0.0000, 0.0000, 0.0000],
        opacity=None, depth=-7.0, interpolate=True)
    
    # --- Initialize components for Routine "reward_screen" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='green', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=8.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Onset_screen" ---
        # create an object to store info about Routine Onset_screen
        Onset_screen = data.Routine(
            name='Onset_screen',
            components=[onset_rect, mouse_3, text],
        )
        Onset_screen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the mouse_3
        mouse_3.x = []
        mouse_3.y = []
        mouse_3.leftButton = []
        mouse_3.midButton = []
        mouse_3.rightButton = []
        mouse_3.time = []
        mouse_3.clicked_name = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_2
        
        mouse.setVisible(1)
        # store start times for Onset_screen
        Onset_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Onset_screen.tStart = globalClock.getTime(format='float')
        Onset_screen.status = STARTED
        thisExp.addData('Onset_screen.started', Onset_screen.tStart)
        Onset_screen.maxDuration = None
        # keep track of which components have finished
        Onset_screenComponents = Onset_screen.components
        for thisComponent in Onset_screen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Onset_screen" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        Onset_screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *onset_rect* updates
            
            # if onset_rect is starting this frame...
            if onset_rect.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                onset_rect.frameNStart = frameN  # exact frame index
                onset_rect.tStart = t  # local t and not account for scr refresh
                onset_rect.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(onset_rect, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'onset_rect.started')
                # update status
                onset_rect.status = STARTED
                onset_rect.setAutoDraw(True)
            
            # if onset_rect is active this frame...
            if onset_rect.status == STARTED:
                # update params
                pass
            # *mouse_3* updates
            
            # if mouse_3 is starting this frame...
            if mouse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_3.frameNStart = frameN  # exact frame index
                mouse_3.tStart = t  # local t and not account for scr refresh
                mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_3.started', t)
                # update status
                mouse_3.status = STARTED
                mouse_3.mouseClock.reset()
                prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
            if mouse_3.status == STARTED:  # only update if started and not finished!
                buttons = mouse_3.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([onset_rect], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse_3):
                                gotValidClick = True
                                mouse_3.clicked_name.append(obj.name)
                        if not gotValidClick:
                            mouse_3.clicked_name.append(None)
                        x, y = mouse_3.getPos()
                        mouse_3.x.append(x)
                        mouse_3.y.append(y)
                        buttons = mouse_3.getPressed()
                        mouse_3.leftButton.append(buttons[0])
                        mouse_3.midButton.append(buttons[1])
                        mouse_3.rightButton.append(buttons[2])
                        mouse_3.time.append(mouse_3.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Onset_screen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Onset_screen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Onset_screen" ---
        for thisComponent in Onset_screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Onset_screen
        Onset_screen.tStop = globalClock.getTime(format='float')
        Onset_screen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Onset_screen.stopped', Onset_screen.tStop)
        # store data for trials (TrialHandler)
        trials.addData('mouse_3.x', mouse_3.x)
        trials.addData('mouse_3.y', mouse_3.y)
        trials.addData('mouse_3.leftButton', mouse_3.leftButton)
        trials.addData('mouse_3.midButton', mouse_3.midButton)
        trials.addData('mouse_3.rightButton', mouse_3.rightButton)
        trials.addData('mouse_3.time', mouse_3.time)
        trials.addData('mouse_3.clicked_name', mouse_3.clicked_name)
        # Run 'End Routine' code from code_2
        mouse.clickReset()
        #win.flip()
        #core.wait(0.1)
        # the Routine "Onset_screen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "buffer" ---
        # create an object to store info about Routine buffer
        buffer = data.Routine(
            name='buffer',
            components=[polygon_2],
        )
        buffer.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for buffer
        buffer.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        buffer.tStart = globalClock.getTime(format='float')
        buffer.status = STARTED
        thisExp.addData('buffer.started', buffer.tStart)
        buffer.maxDuration = None
        # keep track of which components have finished
        bufferComponents = buffer.components
        for thisComponent in buffer.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "buffer" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        buffer.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.2:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon_2* updates
            
            # if polygon_2 is starting this frame...
            if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_2.frameNStart = frameN  # exact frame index
                polygon_2.tStart = t  # local t and not account for scr refresh
                polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_2.started')
                # update status
                polygon_2.status = STARTED
                polygon_2.setAutoDraw(True)
            
            # if polygon_2 is active this frame...
            if polygon_2.status == STARTED:
                # update params
                pass
            
            # if polygon_2 is stopping this frame...
            if polygon_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_2.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_2.tStop = t  # not accounting for scr refresh
                    polygon_2.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_2.stopped')
                    # update status
                    polygon_2.status = FINISHED
                    polygon_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                buffer.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in buffer.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "buffer" ---
        for thisComponent in buffer.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for buffer
        buffer.tStop = globalClock.getTime(format='float')
        buffer.tStopRefresh = tThisFlipGlobal
        thisExp.addData('buffer.stopped', buffer.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if buffer.maxDurationReached:
            routineTimer.addTime(-buffer.maxDuration)
        elif buffer.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.200000)
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[polygon_7, target1_4, target2_4, target3_4, target4_4],
        )
        fixation_cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation_cross
        fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_cross.tStart = globalClock.getTime(format='float')
        fixation_cross.status = STARTED
        thisExp.addData('fixation_cross.started', fixation_cross.tStart)
        fixation_cross.maxDuration = None
        # keep track of which components have finished
        fixation_crossComponents = fixation_cross.components
        for thisComponent in fixation_cross.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_cross" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.7:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon_7* updates
            
            # if polygon_7 is starting this frame...
            if polygon_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_7.frameNStart = frameN  # exact frame index
                polygon_7.tStart = t  # local t and not account for scr refresh
                polygon_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_7.started')
                # update status
                polygon_7.status = STARTED
                polygon_7.setAutoDraw(True)
            
            # if polygon_7 is active this frame...
            if polygon_7.status == STARTED:
                # update params
                pass
            
            # if polygon_7 is stopping this frame...
            if polygon_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_7.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_7.tStop = t  # not accounting for scr refresh
                    polygon_7.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon_7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_7.stopped')
                    # update status
                    polygon_7.status = FINISHED
                    polygon_7.setAutoDraw(False)
            
            # *target1_4* updates
            
            # if target1_4 is starting this frame...
            if target1_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target1_4.frameNStart = frameN  # exact frame index
                target1_4.tStart = t  # local t and not account for scr refresh
                target1_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target1_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target1_4.started')
                # update status
                target1_4.status = STARTED
                target1_4.setAutoDraw(True)
            
            # if target1_4 is active this frame...
            if target1_4.status == STARTED:
                # update params
                pass
            
            # if target1_4 is stopping this frame...
            if target1_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target1_4.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    target1_4.tStop = t  # not accounting for scr refresh
                    target1_4.tStopRefresh = tThisFlipGlobal  # on global time
                    target1_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target1_4.stopped')
                    # update status
                    target1_4.status = FINISHED
                    target1_4.setAutoDraw(False)
            
            # *target2_4* updates
            
            # if target2_4 is starting this frame...
            if target2_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target2_4.frameNStart = frameN  # exact frame index
                target2_4.tStart = t  # local t and not account for scr refresh
                target2_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target2_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target2_4.started')
                # update status
                target2_4.status = STARTED
                target2_4.setAutoDraw(True)
            
            # if target2_4 is active this frame...
            if target2_4.status == STARTED:
                # update params
                pass
            
            # if target2_4 is stopping this frame...
            if target2_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target2_4.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    target2_4.tStop = t  # not accounting for scr refresh
                    target2_4.tStopRefresh = tThisFlipGlobal  # on global time
                    target2_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target2_4.stopped')
                    # update status
                    target2_4.status = FINISHED
                    target2_4.setAutoDraw(False)
            
            # *target3_4* updates
            
            # if target3_4 is starting this frame...
            if target3_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target3_4.frameNStart = frameN  # exact frame index
                target3_4.tStart = t  # local t and not account for scr refresh
                target3_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target3_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target3_4.started')
                # update status
                target3_4.status = STARTED
                target3_4.setAutoDraw(True)
            
            # if target3_4 is active this frame...
            if target3_4.status == STARTED:
                # update params
                pass
            
            # if target3_4 is stopping this frame...
            if target3_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target3_4.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    target3_4.tStop = t  # not accounting for scr refresh
                    target3_4.tStopRefresh = tThisFlipGlobal  # on global time
                    target3_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target3_4.stopped')
                    # update status
                    target3_4.status = FINISHED
                    target3_4.setAutoDraw(False)
            
            # *target4_4* updates
            
            # if target4_4 is starting this frame...
            if target4_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target4_4.frameNStart = frameN  # exact frame index
                target4_4.tStart = t  # local t and not account for scr refresh
                target4_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target4_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target4_4.started')
                # update status
                target4_4.status = STARTED
                target4_4.setAutoDraw(True)
            
            # if target4_4 is active this frame...
            if target4_4.status == STARTED:
                # update params
                pass
            
            # if target4_4 is stopping this frame...
            if target4_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target4_4.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    target4_4.tStop = t  # not accounting for scr refresh
                    target4_4.tStopRefresh = tThisFlipGlobal  # on global time
                    target4_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target4_4.stopped')
                    # update status
                    target4_4.status = FINISHED
                    target4_4.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation_cross.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_cross" ---
        for thisComponent in fixation_cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_cross
        fixation_cross.tStop = globalClock.getTime(format='float')
        fixation_cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.700000)
        
        # --- Prepare to start Routine "trial_error_feedback" ---
        # create an object to store info about Routine trial_error_feedback
        trial_error_feedback = data.Routine(
            name='trial_error_feedback',
            components=[target1, target2, target3, target4, mouse],
        )
        trial_error_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        #mouse = event.Mouse()
        #mouse.setPos([0,0])
        index = block1_count
        target = block1_seq[index]
        print("target", target)
        polygons = [target1,target2,target3, target4]
        polygons[target].fillColor = 'red'
        block1_count+=1
        #disapear_flag = False
        # setup some python lists for storing info about the mouse
        mouse.x = []
        mouse.y = []
        mouse.leftButton = []
        mouse.midButton = []
        mouse.rightButton = []
        mouse.time = []
        gotValidClick = False  # until a click is received
        # store start times for trial_error_feedback
        trial_error_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_error_feedback.tStart = globalClock.getTime(format='float')
        trial_error_feedback.status = STARTED
        thisExp.addData('trial_error_feedback.started', trial_error_feedback.tStart)
        trial_error_feedback.maxDuration = None
        # keep track of which components have finished
        trial_error_feedbackComponents = trial_error_feedback.components
        for thisComponent in trial_error_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_error_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        trial_error_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code
            if mouse.isPressedIn(polygons[target], buttons=[0]):
                print("correct")
                continueRoutine = False
                
            if mouse.getPressed()[0] == 1 and mouse.isPressedIn(polygons[target], buttons=[0]) ==False:
                print("Wrong")
                continueRoutine = False
                
            #if not polygon_3.contains(mouse):
            #    disapear_flag = True
            #if disapear_flag == True:
            #    polygon_3.borderColor = "green"
            
            # *target1* updates
            
            # if target1 is starting this frame...
            if target1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target1.frameNStart = frameN  # exact frame index
                target1.tStart = t  # local t and not account for scr refresh
                target1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target1.started')
                # update status
                target1.status = STARTED
                target1.setAutoDraw(True)
            
            # if target1 is active this frame...
            if target1.status == STARTED:
                # update params
                pass
            
            # *target2* updates
            
            # if target2 is starting this frame...
            if target2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target2.frameNStart = frameN  # exact frame index
                target2.tStart = t  # local t and not account for scr refresh
                target2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target2.started')
                # update status
                target2.status = STARTED
                target2.setAutoDraw(True)
            
            # if target2 is active this frame...
            if target2.status == STARTED:
                # update params
                pass
            
            # *target3* updates
            
            # if target3 is starting this frame...
            if target3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target3.frameNStart = frameN  # exact frame index
                target3.tStart = t  # local t and not account for scr refresh
                target3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target3.started')
                # update status
                target3.status = STARTED
                target3.setAutoDraw(True)
            
            # if target3 is active this frame...
            if target3.status == STARTED:
                # update params
                pass
            
            # *target4* updates
            
            # if target4 is starting this frame...
            if target4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target4.frameNStart = frameN  # exact frame index
                target4.tStart = t  # local t and not account for scr refresh
                target4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target4.started')
                # update status
                target4.status = STARTED
                target4.setAutoDraw(True)
            
            # if target4 is active this frame...
            if target4.status == STARTED:
                # update params
                pass
            # *mouse* updates
            
            # if mouse is starting this frame...
            if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse.frameNStart = frameN  # exact frame index
                mouse.tStart = t  # local t and not account for scr refresh
                mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse.started', t)
                # update status
                mouse.status = STARTED
                mouse.mouseClock.reset()
                prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
            if mouse.status == STARTED:  # only update if started and not finished!
                buttons = mouse.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse.getPos()
                        mouse.x.append(x)
                        mouse.y.append(y)
                        buttons = mouse.getPressed()
                        mouse.leftButton.append(buttons[0])
                        mouse.midButton.append(buttons[1])
                        mouse.rightButton.append(buttons[2])
                        mouse.time.append(mouse.mouseClock.getTime())
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_error_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_error_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_error_feedback" ---
        for thisComponent in trial_error_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_error_feedback
        trial_error_feedback.tStop = globalClock.getTime(format='float')
        trial_error_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_error_feedback.stopped', trial_error_feedback.tStop)
        # Run 'End Routine' code from code
        polygons[target].fillColor = 'white'
        mouse.clickReset()
        #win.flip()
        #core.wait(0.2)
        # store data for trials (TrialHandler)
        trials.addData('mouse.x', mouse.x)
        trials.addData('mouse.y', mouse.y)
        trials.addData('mouse.leftButton', mouse.leftButton)
        trials.addData('mouse.midButton', mouse.midButton)
        trials.addData('mouse.rightButton', mouse.rightButton)
        trials.addData('mouse.time', mouse.time)
        # the Routine "trial_error_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 8.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "interblockscreen" ---
    # create an object to store info about Routine interblockscreen
    interblockscreen = data.Routine(
        name='interblockscreen',
        components=[text_2, mouse_6],
    )
    interblockscreen.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_6
    mouse_6.x = []
    mouse_6.y = []
    mouse_6.leftButton = []
    mouse_6.midButton = []
    mouse_6.rightButton = []
    mouse_6.time = []
    gotValidClick = False  # until a click is received
    # store start times for interblockscreen
    interblockscreen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    interblockscreen.tStart = globalClock.getTime(format='float')
    interblockscreen.status = STARTED
    thisExp.addData('interblockscreen.started', interblockscreen.tStart)
    interblockscreen.maxDuration = None
    # keep track of which components have finished
    interblockscreenComponents = interblockscreen.components
    for thisComponent in interblockscreen.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "interblockscreen" ---
    interblockscreen.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        # *mouse_6* updates
        
        # if mouse_6 is starting this frame...
        if mouse_6.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_6.frameNStart = frameN  # exact frame index
            mouse_6.tStart = t  # local t and not account for scr refresh
            mouse_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('mouse_6.started', t)
            # update status
            mouse_6.status = STARTED
            mouse_6.mouseClock.reset()
            prevButtonState = mouse_6.getPressed()  # if button is down already this ISN'T a new click
        if mouse_6.status == STARTED:  # only update if started and not finished!
            buttons = mouse_6.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_6.getPos()
                    mouse_6.x.append(x)
                    mouse_6.y.append(y)
                    buttons = mouse_6.getPressed()
                    mouse_6.leftButton.append(buttons[0])
                    mouse_6.midButton.append(buttons[1])
                    mouse_6.rightButton.append(buttons[2])
                    mouse_6.time.append(mouse_6.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            interblockscreen.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in interblockscreen.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "interblockscreen" ---
    for thisComponent in interblockscreen.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for interblockscreen
    interblockscreen.tStop = globalClock.getTime(format='float')
    interblockscreen.tStopRefresh = tThisFlipGlobal
    thisExp.addData('interblockscreen.stopped', interblockscreen.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_6.x', mouse_6.x)
    thisExp.addData('mouse_6.y', mouse_6.y)
    thisExp.addData('mouse_6.leftButton', mouse_6.leftButton)
    thisExp.addData('mouse_6.midButton', mouse_6.midButton)
    thisExp.addData('mouse_6.rightButton', mouse_6.rightButton)
    thisExp.addData('mouse_6.time', mouse_6.time)
    thisExp.nextEntry()
    # the Routine "interblockscreen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler2(
        name='trials_2',
        nReps=8.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "Onset_screen" ---
        # create an object to store info about Routine Onset_screen
        Onset_screen = data.Routine(
            name='Onset_screen',
            components=[onset_rect, mouse_3, text],
        )
        Onset_screen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the mouse_3
        mouse_3.x = []
        mouse_3.y = []
        mouse_3.leftButton = []
        mouse_3.midButton = []
        mouse_3.rightButton = []
        mouse_3.time = []
        mouse_3.clicked_name = []
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_2
        
        mouse.setVisible(1)
        # store start times for Onset_screen
        Onset_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Onset_screen.tStart = globalClock.getTime(format='float')
        Onset_screen.status = STARTED
        thisExp.addData('Onset_screen.started', Onset_screen.tStart)
        Onset_screen.maxDuration = None
        # keep track of which components have finished
        Onset_screenComponents = Onset_screen.components
        for thisComponent in Onset_screen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Onset_screen" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        Onset_screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *onset_rect* updates
            
            # if onset_rect is starting this frame...
            if onset_rect.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                onset_rect.frameNStart = frameN  # exact frame index
                onset_rect.tStart = t  # local t and not account for scr refresh
                onset_rect.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(onset_rect, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'onset_rect.started')
                # update status
                onset_rect.status = STARTED
                onset_rect.setAutoDraw(True)
            
            # if onset_rect is active this frame...
            if onset_rect.status == STARTED:
                # update params
                pass
            # *mouse_3* updates
            
            # if mouse_3 is starting this frame...
            if mouse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_3.frameNStart = frameN  # exact frame index
                mouse_3.tStart = t  # local t and not account for scr refresh
                mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_3.started', t)
                # update status
                mouse_3.status = STARTED
                mouse_3.mouseClock.reset()
                prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
            if mouse_3.status == STARTED:  # only update if started and not finished!
                buttons = mouse_3.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames([onset_rect], namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse_3):
                                gotValidClick = True
                                mouse_3.clicked_name.append(obj.name)
                        if not gotValidClick:
                            mouse_3.clicked_name.append(None)
                        x, y = mouse_3.getPos()
                        mouse_3.x.append(x)
                        mouse_3.y.append(y)
                        buttons = mouse_3.getPressed()
                        mouse_3.leftButton.append(buttons[0])
                        mouse_3.midButton.append(buttons[1])
                        mouse_3.rightButton.append(buttons[2])
                        mouse_3.time.append(mouse_3.mouseClock.getTime())
                        if gotValidClick:
                            continueRoutine = False  # end routine on response
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Onset_screen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Onset_screen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Onset_screen" ---
        for thisComponent in Onset_screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Onset_screen
        Onset_screen.tStop = globalClock.getTime(format='float')
        Onset_screen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Onset_screen.stopped', Onset_screen.tStop)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('mouse_3.x', mouse_3.x)
        trials_2.addData('mouse_3.y', mouse_3.y)
        trials_2.addData('mouse_3.leftButton', mouse_3.leftButton)
        trials_2.addData('mouse_3.midButton', mouse_3.midButton)
        trials_2.addData('mouse_3.rightButton', mouse_3.rightButton)
        trials_2.addData('mouse_3.time', mouse_3.time)
        trials_2.addData('mouse_3.clicked_name', mouse_3.clicked_name)
        # Run 'End Routine' code from code_2
        mouse.clickReset()
        #win.flip()
        #core.wait(0.1)
        # the Routine "Onset_screen" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "buffer" ---
        # create an object to store info about Routine buffer
        buffer = data.Routine(
            name='buffer',
            components=[polygon_2],
        )
        buffer.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for buffer
        buffer.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        buffer.tStart = globalClock.getTime(format='float')
        buffer.status = STARTED
        thisExp.addData('buffer.started', buffer.tStart)
        buffer.maxDuration = None
        # keep track of which components have finished
        bufferComponents = buffer.components
        for thisComponent in buffer.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "buffer" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        buffer.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.2:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon_2* updates
            
            # if polygon_2 is starting this frame...
            if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_2.frameNStart = frameN  # exact frame index
                polygon_2.tStart = t  # local t and not account for scr refresh
                polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_2.started')
                # update status
                polygon_2.status = STARTED
                polygon_2.setAutoDraw(True)
            
            # if polygon_2 is active this frame...
            if polygon_2.status == STARTED:
                # update params
                pass
            
            # if polygon_2 is stopping this frame...
            if polygon_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_2.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_2.tStop = t  # not accounting for scr refresh
                    polygon_2.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_2.stopped')
                    # update status
                    polygon_2.status = FINISHED
                    polygon_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                buffer.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in buffer.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "buffer" ---
        for thisComponent in buffer.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for buffer
        buffer.tStop = globalClock.getTime(format='float')
        buffer.tStopRefresh = tThisFlipGlobal
        thisExp.addData('buffer.stopped', buffer.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if buffer.maxDurationReached:
            routineTimer.addTime(-buffer.maxDuration)
        elif buffer.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.200000)
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[polygon_7, target1_4, target2_4, target3_4, target4_4],
        )
        fixation_cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fixation_cross
        fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_cross.tStart = globalClock.getTime(format='float')
        fixation_cross.status = STARTED
        thisExp.addData('fixation_cross.started', fixation_cross.tStart)
        fixation_cross.maxDuration = None
        # keep track of which components have finished
        fixation_crossComponents = fixation_cross.components
        for thisComponent in fixation_cross.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_cross" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.7:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon_7* updates
            
            # if polygon_7 is starting this frame...
            if polygon_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_7.frameNStart = frameN  # exact frame index
                polygon_7.tStart = t  # local t and not account for scr refresh
                polygon_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_7.started')
                # update status
                polygon_7.status = STARTED
                polygon_7.setAutoDraw(True)
            
            # if polygon_7 is active this frame...
            if polygon_7.status == STARTED:
                # update params
                pass
            
            # if polygon_7 is stopping this frame...
            if polygon_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_7.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_7.tStop = t  # not accounting for scr refresh
                    polygon_7.tStopRefresh = tThisFlipGlobal  # on global time
                    polygon_7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'polygon_7.stopped')
                    # update status
                    polygon_7.status = FINISHED
                    polygon_7.setAutoDraw(False)
            
            # *target1_4* updates
            
            # if target1_4 is starting this frame...
            if target1_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target1_4.frameNStart = frameN  # exact frame index
                target1_4.tStart = t  # local t and not account for scr refresh
                target1_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target1_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target1_4.started')
                # update status
                target1_4.status = STARTED
                target1_4.setAutoDraw(True)
            
            # if target1_4 is active this frame...
            if target1_4.status == STARTED:
                # update params
                pass
            
            # if target1_4 is stopping this frame...
            if target1_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target1_4.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    target1_4.tStop = t  # not accounting for scr refresh
                    target1_4.tStopRefresh = tThisFlipGlobal  # on global time
                    target1_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target1_4.stopped')
                    # update status
                    target1_4.status = FINISHED
                    target1_4.setAutoDraw(False)
            
            # *target2_4* updates
            
            # if target2_4 is starting this frame...
            if target2_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target2_4.frameNStart = frameN  # exact frame index
                target2_4.tStart = t  # local t and not account for scr refresh
                target2_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target2_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target2_4.started')
                # update status
                target2_4.status = STARTED
                target2_4.setAutoDraw(True)
            
            # if target2_4 is active this frame...
            if target2_4.status == STARTED:
                # update params
                pass
            
            # if target2_4 is stopping this frame...
            if target2_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target2_4.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    target2_4.tStop = t  # not accounting for scr refresh
                    target2_4.tStopRefresh = tThisFlipGlobal  # on global time
                    target2_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target2_4.stopped')
                    # update status
                    target2_4.status = FINISHED
                    target2_4.setAutoDraw(False)
            
            # *target3_4* updates
            
            # if target3_4 is starting this frame...
            if target3_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target3_4.frameNStart = frameN  # exact frame index
                target3_4.tStart = t  # local t and not account for scr refresh
                target3_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target3_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target3_4.started')
                # update status
                target3_4.status = STARTED
                target3_4.setAutoDraw(True)
            
            # if target3_4 is active this frame...
            if target3_4.status == STARTED:
                # update params
                pass
            
            # if target3_4 is stopping this frame...
            if target3_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target3_4.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    target3_4.tStop = t  # not accounting for scr refresh
                    target3_4.tStopRefresh = tThisFlipGlobal  # on global time
                    target3_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target3_4.stopped')
                    # update status
                    target3_4.status = FINISHED
                    target3_4.setAutoDraw(False)
            
            # *target4_4* updates
            
            # if target4_4 is starting this frame...
            if target4_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target4_4.frameNStart = frameN  # exact frame index
                target4_4.tStart = t  # local t and not account for scr refresh
                target4_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target4_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target4_4.started')
                # update status
                target4_4.status = STARTED
                target4_4.setAutoDraw(True)
            
            # if target4_4 is active this frame...
            if target4_4.status == STARTED:
                # update params
                pass
            
            # if target4_4 is stopping this frame...
            if target4_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target4_4.tStartRefresh + 0.7-frameTolerance:
                    # keep track of stop time/frame for later
                    target4_4.tStop = t  # not accounting for scr refresh
                    target4_4.tStopRefresh = tThisFlipGlobal  # on global time
                    target4_4.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target4_4.stopped')
                    # update status
                    target4_4.status = FINISHED
                    target4_4.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation_cross.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_cross" ---
        for thisComponent in fixation_cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_cross
        fixation_cross.tStop = globalClock.getTime(format='float')
        fixation_cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.700000)
        
        # --- Prepare to start Routine "trial_reward_feedback" ---
        # create an object to store info about Routine trial_reward_feedback
        trial_reward_feedback = data.Routine(
            name='trial_reward_feedback',
            components=[target1_2, target2_2, target3_2, target4_2, mouse_4, polygon_4],
        )
        trial_reward_feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_3
        index = block2_count
        target = block2_seq[index]
        print("target", target)
        polygons = [target1_2,target2_2,target3_2, target4_2]
        for p in polygons:
            p.borderColor = 'white'
            p.fillColor = 'gray'
        polygons[target].fillColor = 'red'
        block2_count+=1
        
        correct_flag = False
        wrong_flag = False
        
        #disapear_flag = False
        # Run 'Begin Routine' code from mouse_tracking
        disapear_flag = False
        mouse.setVisible(1)
        # setup some python lists for storing info about the mouse_4
        mouse_4.x = []
        mouse_4.y = []
        mouse_4.leftButton = []
        mouse_4.midButton = []
        mouse_4.rightButton = []
        mouse_4.time = []
        gotValidClick = False  # until a click is received
        # store start times for trial_reward_feedback
        trial_reward_feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_reward_feedback.tStart = globalClock.getTime(format='float')
        trial_reward_feedback.status = STARTED
        thisExp.addData('trial_reward_feedback.started', trial_reward_feedback.tStart)
        trial_reward_feedback.maxDuration = None
        # keep track of which components have finished
        trial_reward_feedbackComponents = trial_reward_feedback.components
        for thisComponent in trial_reward_feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_reward_feedback" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        trial_reward_feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_3
            if mouse.isPressedIn(polygons[target], buttons=[0]):
                print("correct")
                correct_flag = True
                continueRoutine = False
                
            if mouse.getPressed()[0] == 1 and mouse.isPressedIn(polygons[target], buttons=[0]) ==False:
                print("Wrong")
                continueRoutine = False
                wrong_flag = True
                
            #if not polygon_4.contains(mouse) and disapear_flag == False:
            #    disapear_flag = True
            #    mouse.setVisible(0)
            #    polygon_4.borderColor = "green"
            #    for p in polygons:
            #        p.borderColor = 'gray'
            #        p.fillColor = 'gray'
            # Run 'Each Frame' code from mouse_tracking
            if not polygon_4.contains(mouse) and disapear_flag == False:
                disapear_flag = True
                mouse.setVisible(0)
                polygon_4.borderColor = "green"
                for p in polygons:
                    p.borderColor = 'gray'
                    p.fillColor = 'gray'
            
            # *target1_2* updates
            
            # if target1_2 is starting this frame...
            if target1_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target1_2.frameNStart = frameN  # exact frame index
                target1_2.tStart = t  # local t and not account for scr refresh
                target1_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target1_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target1_2.started')
                # update status
                target1_2.status = STARTED
                target1_2.setAutoDraw(True)
            
            # if target1_2 is active this frame...
            if target1_2.status == STARTED:
                # update params
                pass
            
            # *target2_2* updates
            
            # if target2_2 is starting this frame...
            if target2_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target2_2.frameNStart = frameN  # exact frame index
                target2_2.tStart = t  # local t and not account for scr refresh
                target2_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target2_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target2_2.started')
                # update status
                target2_2.status = STARTED
                target2_2.setAutoDraw(True)
            
            # if target2_2 is active this frame...
            if target2_2.status == STARTED:
                # update params
                pass
            
            # *target3_2* updates
            
            # if target3_2 is starting this frame...
            if target3_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target3_2.frameNStart = frameN  # exact frame index
                target3_2.tStart = t  # local t and not account for scr refresh
                target3_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target3_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target3_2.started')
                # update status
                target3_2.status = STARTED
                target3_2.setAutoDraw(True)
            
            # if target3_2 is active this frame...
            if target3_2.status == STARTED:
                # update params
                pass
            
            # *target4_2* updates
            
            # if target4_2 is starting this frame...
            if target4_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target4_2.frameNStart = frameN  # exact frame index
                target4_2.tStart = t  # local t and not account for scr refresh
                target4_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target4_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target4_2.started')
                # update status
                target4_2.status = STARTED
                target4_2.setAutoDraw(True)
            
            # if target4_2 is active this frame...
            if target4_2.status == STARTED:
                # update params
                pass
            # *mouse_4* updates
            
            # if mouse_4 is starting this frame...
            if mouse_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_4.frameNStart = frameN  # exact frame index
                mouse_4.tStart = t  # local t and not account for scr refresh
                mouse_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_4.started', t)
                # update status
                mouse_4.status = STARTED
                mouse_4.mouseClock.reset()
                prevButtonState = mouse_4.getPressed()  # if button is down already this ISN'T a new click
            if mouse_4.status == STARTED:  # only update if started and not finished!
                buttons = mouse_4.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        pass
                        x, y = mouse_4.getPos()
                        mouse_4.x.append(x)
                        mouse_4.y.append(y)
                        buttons = mouse_4.getPressed()
                        mouse_4.leftButton.append(buttons[0])
                        mouse_4.midButton.append(buttons[1])
                        mouse_4.rightButton.append(buttons[2])
                        mouse_4.time.append(mouse_4.mouseClock.getTime())
            
            # *polygon_4* updates
            
            # if polygon_4 is starting this frame...
            if polygon_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_4.frameNStart = frameN  # exact frame index
                polygon_4.tStart = t  # local t and not account for scr refresh
                polygon_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'polygon_4.started')
                # update status
                polygon_4.status = STARTED
                polygon_4.setAutoDraw(True)
            
            # if polygon_4 is active this frame...
            if polygon_4.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_reward_feedback.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_reward_feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_reward_feedback" ---
        for thisComponent in trial_reward_feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_reward_feedback
        trial_reward_feedback.tStop = globalClock.getTime(format='float')
        trial_reward_feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_reward_feedback.stopped', trial_reward_feedback.tStop)
        # Run 'End Routine' code from code_3
        #polygons[target].fillColor = 'white'
        mouse.clickReset()
        #win.flip()
        #core.wait(0.2)
        print('Correct', correct_flag)
        print('Wrong', wrong_flag)
        # Run 'End Routine' code from mouse_tracking
        mouse.setVisible(1)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('mouse_4.x', mouse_4.x)
        trials_2.addData('mouse_4.y', mouse_4.y)
        trials_2.addData('mouse_4.leftButton', mouse_4.leftButton)
        trials_2.addData('mouse_4.midButton', mouse_4.midButton)
        trials_2.addData('mouse_4.rightButton', mouse_4.rightButton)
        trials_2.addData('mouse_4.time', mouse_4.time)
        # the Routine "trial_reward_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "reward_screen" ---
        # create an object to store info about Routine reward_screen
        reward_screen = data.Routine(
            name='reward_screen',
            components=[text_3],
        )
        reward_screen.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_5
        if correct_flag == True:
            stim = visual.TextStim(win, 'Félicitation, vous avez réussi!!', color=(0, 1, 0), colorSpace='rgb')
            
        if wrong_flag == True:
            stim = visual.TextStim(win, 'Vous avez échoué', color=(1, 0, 0), colorSpace='rgb')
        
        # store start times for reward_screen
        reward_screen.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        reward_screen.tStart = globalClock.getTime(format='float')
        reward_screen.status = STARTED
        thisExp.addData('reward_screen.started', reward_screen.tStart)
        reward_screen.maxDuration = None
        # keep track of which components have finished
        reward_screenComponents = reward_screen.components
        for thisComponent in reward_screen.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "reward_screen" ---
        # if trial has changed, end Routine now
        if isinstance(trials_2, data.TrialHandler2) and thisTrial_2.thisN != trials_2.thisTrial.thisN:
            continueRoutine = False
        reward_screen.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.started')
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # if text_3 is stopping this frame...
            if text_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_3.tStartRefresh + 1-frameTolerance:
                    # keep track of stop time/frame for later
                    text_3.tStop = t  # not accounting for scr refresh
                    text_3.tStopRefresh = tThisFlipGlobal  # on global time
                    text_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_3.stopped')
                    # update status
                    text_3.status = FINISHED
                    text_3.setAutoDraw(False)
            # Run 'Each Frame' code from code_5
            stim.draw()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                reward_screen.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reward_screen.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reward_screen" ---
        for thisComponent in reward_screen.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for reward_screen
        reward_screen.tStop = globalClock.getTime(format='float')
        reward_screen.tStopRefresh = tThisFlipGlobal
        thisExp.addData('reward_screen.stopped', reward_screen.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if reward_screen.maxDurationReached:
            routineTimer.addTime(-reward_screen.maxDuration)
        elif reward_screen.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        thisExp.nextEntry()
        
    # completed 8.0 repeats of 'trials_2'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
