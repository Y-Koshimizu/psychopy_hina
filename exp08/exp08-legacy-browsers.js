/************** 
 * Exp08 *
 **************/


// store info about the experiment session:
let expName = 'exp08';  // from the Builder filename that created this script
let expInfo = {
    'participant': `${util.pad(Number.parseFloat(util.randint(0, 999999)).toFixed(0), 6)}`,
    'session': '001',
};

// Start code blocks for 'Before Experiment'
// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0,0,0]),
  units: 'height',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); }, flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);



flowScheduler.add(quitPsychoJS, '', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, '', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.EXP);


var currentLoop;
var frameDur;
async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2023.2.3';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}


var readyClock;
var mouseReady;
var startDisc;
var probeReady;
var text;
var trialClock;
var mouseTrial;
var path1;
var path2;
var path3;
var path4;
var path5;
var goalDisc;
var probe;
var globalClock;
var routineTimer;
async function experimentInit() {
  // Initialize components for Routine "ready"
  readyClock = new util.Clock();
  mouseReady = new core.Mouse({
    win: psychoJS.window,
  });
  mouseReady.mouseClock = new util.Clock();
  startDisc = new visual.Polygon({
    win: psychoJS.window, name: 'startDisc', 
    edges: 100, size:[0.03, 0.03],
    ori: 0.0, pos: [0, 0],
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('green'),
    opacity: undefined, depth: -1, interpolate: true,
  });
  
  probeReady = new visual.Polygon({
    win: psychoJS.window, name: 'probeReady', 
    edges: 100, size:[0.01, 0.01],
    ori: 0.0, pos: [0, 0],
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('red'),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  text = new visual.TextStim({
    win: psychoJS.window,
    name: 'text',
    text: '準備ができたらマウスの左ボタンをクリックしてください',
    font: 'Open Sans',
    units: undefined, 
    pos: [0, (- 0.32)], height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  mouseTrial = new core.Mouse({
    win: psychoJS.window,
  });
  mouseTrial.mouseClock = new util.Clock();
  // Run 'Begin Experiment' code from codeTrial
  mouseTrial.setVisible(false);
  mouseReady.setVisible(false);
  
  path1 = new visual.Rect ({
    win: psychoJS.window, name: 'path1', 
    width: [0.24, 0.02][0], height: [0.24, 0.02][1],
    ori: 1.0, pos: [0, 0],
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('white'),
    opacity: undefined, depth: -2, interpolate: true,
  });
  
  path2 = new visual.Rect ({
    win: psychoJS.window, name: 'path2', 
    width: [0.24, 0.02][0], height: [0.24, 0.02][1],
    ori: 1.0, pos: [0, 0],
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('white'),
    opacity: undefined, depth: -3, interpolate: true,
  });
  
  path3 = new visual.Rect ({
    win: psychoJS.window, name: 'path3', 
    width: [0.24, 0.02][0], height: [0.24, 0.02][1],
    ori: 1.0, pos: [0, 0],
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('white'),
    opacity: undefined, depth: -4, interpolate: true,
  });
  
  path4 = new visual.Rect ({
    win: psychoJS.window, name: 'path4', 
    width: [0.24, 0.02][0], height: [0.24, 0.02][1],
    ori: 1.0, pos: [0, 0],
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('white'),
    opacity: undefined, depth: -5, interpolate: true,
  });
  
  path5 = new visual.Rect ({
    win: psychoJS.window, name: 'path5', 
    width: [0.24, 0.02][0], height: [0.24, 0.02][1],
    ori: 1.0, pos: [0, 0],
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('white'),
    opacity: undefined, depth: -6, interpolate: true,
  });
  
  goalDisc = new visual.Polygon({
    win: psychoJS.window, name: 'goalDisc', 
    edges: 100, size:[0.03, 0.03],
    ori: 0.0, pos: [0, 0],
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('green'),
    opacity: undefined, depth: -7, interpolate: true,
  });
  
  probe = new visual.Polygon({
    win: psychoJS.window, name: 'probe', 
    edges: 100, size:[0.01, 0.01],
    ori: 0.0, pos: [0, 0],
    anchor: 'center',
    lineWidth: 1.0, 
    colorSpace: 'rgb',
    lineColor: new util.Color('white'),
    fillColor: new util.Color('red'),
    opacity: undefined, depth: -8, interpolate: true,
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var trials;
function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 1, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: 'exp08cnd.xlsx',
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    trials.forEach(function() {
      snapshot = trials.getSnapshot();
    
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(readyRoutineBegin(snapshot));
      trialsLoopScheduler.add(readyRoutineEachFrame());
      trialsLoopScheduler.add(readyRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialRoutineBegin(snapshot));
      trialsLoopScheduler.add(trialRoutineEachFrame());
      trialsLoopScheduler.add(trialRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    });
    
    return Scheduler.Event.NEXT;
  }
}


async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      } else {
        psychoJS.experiment.nextEntry(snapshot);
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var t;
var frameN;
var continueRoutine;
var gotValidClick;
var readyComponents;
function readyRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'ready' ---
    t = 0;
    readyClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    psychoJS.experiment.addData('ready.started', globalClock.getTime());
    // setup some python lists for storing info about the mouseReady
    gotValidClick = false; // until a click is received
    startDisc.setPos(startPos);
    probeReady.setPos(startPos);
    // keep track of which components have finished
    readyComponents = [];
    readyComponents.push(mouseReady);
    readyComponents.push(startDisc);
    readyComponents.push(probeReady);
    readyComponents.push(text);
    
    readyComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


var prevButtonState;
var _mouseButtons;
function readyRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'ready' ---
    // get current time
    t = readyClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // *mouseReady* updates
    if (t >= 0.0 && mouseReady.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouseReady.tStart = t;  // (not accounting for frame time here)
      mouseReady.frameNStart = frameN;  // exact frame index
      
      mouseReady.status = PsychoJS.Status.STARTED;
      mouseReady.mouseClock.reset();
      prevButtonState = mouseReady.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouseReady.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      _mouseButtons = mouseReady.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // end routine on response
          continueRoutine = false;
        }
      }
    }
    
    // *startDisc* updates
    if (t >= 0.0 && startDisc.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      startDisc.tStart = t;  // (not accounting for frame time here)
      startDisc.frameNStart = frameN;  // exact frame index
      
      startDisc.setAutoDraw(true);
    }
    
    
    // *probeReady* updates
    if (t >= 0.0 && probeReady.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      probeReady.tStart = t;  // (not accounting for frame time here)
      probeReady.frameNStart = frameN;  // exact frame index
      
      probeReady.setAutoDraw(true);
    }
    
    
    // *text* updates
    if (t >= 0.0 && text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text.tStart = t;  // (not accounting for frame time here)
      text.frameNStart = frameN;  // exact frame index
      
      text.setAutoDraw(true);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    readyComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function readyRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'ready' ---
    readyComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('ready.stopped', globalClock.getTime());
    // store data for psychoJS.experiment (ExperimentHandler)
    // the Routine "ready" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var probeX_list;
var probeY_list;
var onPath_list;
var frameN_list;
var trialComponents;
function trialRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'trial' ---
    t = 0;
    trialClock.reset(); // clock
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    // update component parameters for each repeat
    psychoJS.experiment.addData('trial.started', globalClock.getTime());
    // setup some python lists for storing info about the mouseTrial
    gotValidClick = false; // until a click is received
    // Run 'Begin Routine' code from codeTrial
    mouseTrial.setPos([startPos[0], (- startPos[1])]);
    probeX_list = [];
    probeY_list = [];
    onPath_list = [];
    frameN_list = [];
    
    path1.setPos(path1pos);
    path1.setOri(path1ori);
    path2.setPos(path2pos);
    path2.setOri(path2ori);
    path3.setPos(path3pos);
    path3.setOri(path3ori);
    path4.setPos(path4pos);
    path4.setOri(path4ori);
    path5.setPos(path5pos);
    path5.setOri(path5ori);
    goalDisc.setPos(goalPos);
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(mouseTrial);
    trialComponents.push(path1);
    trialComponents.push(path2);
    trialComponents.push(path3);
    trialComponents.push(path4);
    trialComponents.push(path5);
    trialComponents.push(goalDisc);
    trialComponents.push(probe);
    
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
       });
    return Scheduler.Event.NEXT;
  }
}


var mousePos;
var px;
var py;
var onPath;
function trialRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'trial' ---
    // get current time
    t = trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // Run 'Each Frame' code from codeTrial
    mousePos = mouseTrial.getPos();
    px = mousePos[0];
    py = (- mousePos[1]);
    if (goalDisc.contains([px, py])) {
        continueRoutine = false;
    }
    onPath = false;
    for (var path, _pj_c = 0, _pj_a = [path1, path2, path3, path4, path5], _pj_b = _pj_a.length; (_pj_c < _pj_b); _pj_c += 1) {
        path = _pj_a[_pj_c];
        if (path.contains([px, py])) {
            onPath = true;
            break;
        }
    }
    if (((frameN % 6) === 0)) {
        probeX_list.push(px);
        probeY_list.push(py);
        onPath_list.push(onPath);
    }
    
    
    // *path1* updates
    if (t >= 0.0 && path1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      path1.tStart = t;  // (not accounting for frame time here)
      path1.frameNStart = frameN;  // exact frame index
      
      path1.setAutoDraw(true);
    }
    
    
    // *path2* updates
    if (t >= 0.0 && path2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      path2.tStart = t;  // (not accounting for frame time here)
      path2.frameNStart = frameN;  // exact frame index
      
      path2.setAutoDraw(true);
    }
    
    
    // *path3* updates
    if (t >= 0.0 && path3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      path3.tStart = t;  // (not accounting for frame time here)
      path3.frameNStart = frameN;  // exact frame index
      
      path3.setAutoDraw(true);
    }
    
    
    // *path4* updates
    if (t >= 0.0 && path4.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      path4.tStart = t;  // (not accounting for frame time here)
      path4.frameNStart = frameN;  // exact frame index
      
      path4.setAutoDraw(true);
    }
    
    
    // *path5* updates
    if (t >= 0.0 && path5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      path5.tStart = t;  // (not accounting for frame time here)
      path5.frameNStart = frameN;  // exact frame index
      
      path5.setAutoDraw(true);
    }
    
    
    // *goalDisc* updates
    if (t >= 0.0 && goalDisc.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      goalDisc.tStart = t;  // (not accounting for frame time here)
      goalDisc.frameNStart = frameN;  // exact frame index
      
      goalDisc.setAutoDraw(true);
    }
    
    
    if (probe.status === PsychoJS.Status.STARTED){ // only update if being drawn
      probe.setPos([px, py], false);
    }
    
    // *probe* updates
    if (t >= 0.0 && probe.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      probe.tStart = t;  // (not accounting for frame time here)
      probe.frameNStart = frameN;  // exact frame index
      
      probe.setAutoDraw(true);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    trialComponents.forEach( function(thisComponent) {
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
      }
    });
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function trialRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'trial' ---
    trialComponents.forEach( function(thisComponent) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    });
    psychoJS.experiment.addData('trial.stopped', globalClock.getTime());
    // store data for psychoJS.experiment (ExperimentHandler)
    // Run 'End Routine' code from codeTrial
    trials.addData("probe_x", probeX_list);
    trials.addData("probe_y", probeY_list);
    trials.addData("on_path", onPath_list);
    trials.addData("frameN", frameN_list);
    
    // the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  
  
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
