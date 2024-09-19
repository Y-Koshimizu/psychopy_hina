#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on 9月 19, 2024, at 17:23
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
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
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

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'exp08'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
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
        originPath='C:\\Users\\user\\Documents\\PsychoPy\\2024\\hina\\exp08\\exp08_lastrun.py',
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
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
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
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1600, 900], fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
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
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='event')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
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
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='Pyglet')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
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
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
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
    
    # --- Initialize components for Routine "ready" ---
    mouseReady = event.Mouse(win=win)
    x, y = [None, None]
    mouseReady.mouseClock = core.Clock()
    startDisc = visual.ShapeStim(
        win=win, name='startDisc',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='green',
        opacity=None, depth=-1.0, interpolate=True)
    probeReady = visual.ShapeStim(
        win=win, name='probeReady',
        size=(0.01, 0.01), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-2.0, interpolate=True)
    text = visual.TextStim(win=win, name='text',
        text='準備ができたらマウスの左ボタンをクリックしてください',
        font='Open Sans',
        pos=[0, -0.32], height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "trial" ---
    mouseTrial = event.Mouse(win=win)
    x, y = [None, None]
    mouseTrial.mouseClock = core.Clock()
    # Run 'Begin Experiment' code from codeTrial
    mouseTrial.setVisible(False)
    mouseReady.setVisible(False)
    path1 = visual.Rect(
        win=win, name='path1',
        width=(0.24, 0.02)[0], height=(0.24, 0.02)[1],
        ori=1.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    path2 = visual.Rect(
        win=win, name='path2',
        width=(0.24, 0.02)[0], height=(0.24, 0.02)[1],
        ori=1.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    path3 = visual.Rect(
        win=win, name='path3',
        width=(0.24, 0.02)[0], height=(0.24, 0.02)[1],
        ori=1.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    path4 = visual.Rect(
        win=win, name='path4',
        width=(0.24, 0.02)[0], height=(0.24, 0.02)[1],
        ori=1.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    path5 = visual.Rect(
        win=win, name='path5',
        width=(0.24, 0.02)[0], height=(0.24, 0.02)[1],
        ori=1.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-6.0, interpolate=True)
    goalDisc = visual.ShapeStim(
        win=win, name='goalDisc',
        size=(0.03, 0.03), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='green',
        opacity=None, depth=-7.0, interpolate=True)
    probe = visual.ShapeStim(
        win=win, name='probe',
        size=(0.01, 0.01), vertices='circle',
        ori=0.0, pos=[0,0], anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='red',
        opacity=None, depth=-8.0, interpolate=True)
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('exp08cnd.xlsx'),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "ready" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('ready.started', globalClock.getTime())
        # setup some python lists for storing info about the mouseReady
        gotValidClick = False  # until a click is received
        startDisc.setPos(startPos)
        probeReady.setPos(startPos)
        # keep track of which components have finished
        readyComponents = [mouseReady, startDisc, probeReady, text]
        for thisComponent in readyComponents:
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
        
        # --- Run Routine "ready" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *mouseReady* updates
            
            # if mouseReady is starting this frame...
            if mouseReady.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouseReady.frameNStart = frameN  # exact frame index
                mouseReady.tStart = t  # local t and not account for scr refresh
                mouseReady.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouseReady, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouseReady.started', t)
                # update status
                mouseReady.status = STARTED
                mouseReady.mouseClock.reset()
                prevButtonState = mouseReady.getPressed()  # if button is down already this ISN'T a new click
            if mouseReady.status == STARTED:  # only update if started and not finished!
                buttons = mouseReady.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        continueRoutine = False  # end routine on response            
            # *startDisc* updates
            
            # if startDisc is starting this frame...
            if startDisc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                startDisc.frameNStart = frameN  # exact frame index
                startDisc.tStart = t  # local t and not account for scr refresh
                startDisc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(startDisc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'startDisc.started')
                # update status
                startDisc.status = STARTED
                startDisc.setAutoDraw(True)
            
            # if startDisc is active this frame...
            if startDisc.status == STARTED:
                # update params
                pass
            
            # *probeReady* updates
            
            # if probeReady is starting this frame...
            if probeReady.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                probeReady.frameNStart = frameN  # exact frame index
                probeReady.tStart = t  # local t and not account for scr refresh
                probeReady.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probeReady, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probeReady.started')
                # update status
                probeReady.status = STARTED
                probeReady.setAutoDraw(True)
            
            # if probeReady is active this frame...
            if probeReady.status == STARTED:
                # update params
                pass
            
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
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in readyComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ready" ---
        for thisComponent in readyComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('ready.stopped', globalClock.getTime())
        # store data for trials (TrialHandler)
        # the Routine "ready" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        # setup some python lists for storing info about the mouseTrial
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from codeTrial
        mouseTrial.setPos([startPos[0], -startPos[1]])
        probeX_list = [ ]
        probeY_list = [ ]
        onPath_list = [ ]
        
        path1.setPos(path1pos)
        path1.setOri(path1ori)
        path2.setPos(path2pos)
        path2.setOri(path2ori)
        path3.setPos(path3pos)
        path3.setOri(path3ori)
        path4.setPos(path4pos)
        path4.setOri(path4ori)
        path5.setPos(path5pos)
        path5.setOri(path5ori)
        goalDisc.setPos(goalPos)
        # keep track of which components have finished
        trialComponents = [mouseTrial, path1, path2, path3, path4, path5, goalDisc, probe]
        for thisComponent in trialComponents:
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
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *mouseTrial* updates
            
            # if mouseTrial is starting this frame...
            if mouseTrial.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouseTrial.frameNStart = frameN  # exact frame index
                mouseTrial.tStart = t  # local t and not account for scr refresh
                mouseTrial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouseTrial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouseTrial.started', t)
                # update status
                mouseTrial.status = STARTED
                mouseTrial.mouseClock.reset()
                prevButtonState = [0, 0, 0]  # if now button is down we will treat as 'new' click
            # Run 'Each Frame' code from codeTrial
            mousePos = mouseTrial.getPos()
            px = mousePos[0]
            py = -mousePos[1]
            if goalDisc.contains([px, py]):
                continueRoutine = False
            onPath = False
            for path in [path1, path2, path3, path4, path5]:
                if path.contains([px, py]):
                    onPath = True
                    break
            probeX_list.append(px)
            probeY_list.append(py)
            onPath_list.append(onPath)
            
            # *path1* updates
            
            # if path1 is starting this frame...
            if path1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                path1.frameNStart = frameN  # exact frame index
                path1.tStart = t  # local t and not account for scr refresh
                path1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(path1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'path1.started')
                # update status
                path1.status = STARTED
                path1.setAutoDraw(True)
            
            # if path1 is active this frame...
            if path1.status == STARTED:
                # update params
                pass
            
            # *path2* updates
            
            # if path2 is starting this frame...
            if path2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                path2.frameNStart = frameN  # exact frame index
                path2.tStart = t  # local t and not account for scr refresh
                path2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(path2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'path2.started')
                # update status
                path2.status = STARTED
                path2.setAutoDraw(True)
            
            # if path2 is active this frame...
            if path2.status == STARTED:
                # update params
                pass
            
            # *path3* updates
            
            # if path3 is starting this frame...
            if path3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                path3.frameNStart = frameN  # exact frame index
                path3.tStart = t  # local t and not account for scr refresh
                path3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(path3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'path3.started')
                # update status
                path3.status = STARTED
                path3.setAutoDraw(True)
            
            # if path3 is active this frame...
            if path3.status == STARTED:
                # update params
                pass
            
            # *path4* updates
            
            # if path4 is starting this frame...
            if path4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                path4.frameNStart = frameN  # exact frame index
                path4.tStart = t  # local t and not account for scr refresh
                path4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(path4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'path4.started')
                # update status
                path4.status = STARTED
                path4.setAutoDraw(True)
            
            # if path4 is active this frame...
            if path4.status == STARTED:
                # update params
                pass
            
            # *path5* updates
            
            # if path5 is starting this frame...
            if path5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                path5.frameNStart = frameN  # exact frame index
                path5.tStart = t  # local t and not account for scr refresh
                path5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(path5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'path5.started')
                # update status
                path5.status = STARTED
                path5.setAutoDraw(True)
            
            # if path5 is active this frame...
            if path5.status == STARTED:
                # update params
                pass
            
            # *goalDisc* updates
            
            # if goalDisc is starting this frame...
            if goalDisc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                goalDisc.frameNStart = frameN  # exact frame index
                goalDisc.tStart = t  # local t and not account for scr refresh
                goalDisc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(goalDisc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'goalDisc.started')
                # update status
                goalDisc.status = STARTED
                goalDisc.setAutoDraw(True)
            
            # if goalDisc is active this frame...
            if goalDisc.status == STARTED:
                # update params
                pass
            
            # *probe* updates
            
            # if probe is starting this frame...
            if probe.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                probe.frameNStart = frameN  # exact frame index
                probe.tStart = t  # local t and not account for scr refresh
                probe.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe.started')
                # update status
                probe.status = STARTED
                probe.setAutoDraw(True)
            
            # if probe is active this frame...
            if probe.status == STARTED:
                # update params
                probe.setPos([px, py], log=False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime())
        # store data for trials (TrialHandler)
        # Run 'End Routine' code from codeTrial
        trials.addData('probe_x', probeX_list)
        trials.addData('probe_y', probeY_list)
        trials.addData('on_path', onPath_list)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials'
    
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


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


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
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
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
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
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
