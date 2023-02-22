# Assignment 2: GStreamer ! Application development: video mixer

In this assignment, you are going to make a rudimentary version of a device capable of:
1. crossfading between 2 video channels;
2. adding a video effect afterwards (toggling between 2 options of effects);
3. adding a logo upon request;
4. displaying the output live;
5. storing the output to an MKV file. 

In a television studio, there needs to be a possibility to fade between different videostreams (camera’s). An example of a video crossfade is displayed below:

![fade](docs/fade.gif)

Additionally, we need to be able to add an effect to the video. The effect can be activated the entire time, but there needs to be a possibility to swap between two effects, though.

As a last visual effect, a logo should be displayed on the screen. It should be possible to activate and deactivate the logo.

This assignment uses a `CMakeLists.txt` file and `CMake` to facilitate the compilation and building of the project. Check out the last section for instructions on how to use CMake.

Questions or remarks are welcome at glenn.vanwallendael@ugent.be with in CC julie.artois@ugent.be and hannes.mareen@ugent.be .

## Tutorials

Please follow the [tutorials](https://gstreamer.freedesktop.org/documentation/tutorials/basic/index.html?gi-language=c) to get you started:
1. Start with a [Hello World](https://gstreamer.freedesktop.org/documentation/tutorials/basic/hello-world.html?gi-language=c) example.
2. Get to know the [GStreamer concepts](https://gstreamer.freedesktop.org/documentation/tutorials/basic/concepts.html?gi-language=c).
3. Knowledge on [dynamic linking](https://gstreamer.freedesktop.org/documentation/tutorials/basic/dynamic-pipelines.html?gi-language=c) is necessary to complete the assignment.
4. Finally, knowledge on [replacing elements with pad probes](https://gstreamer.freedesktop.org/documentation/application-development/advanced/pipeline-manipulation.html?gi-language=c#changing-elements-in-a-pipeline) is necessary to complete the assignment.

## Explanation of mixer.c base file

We will explore the concepts required to allow real-time pipeline alterations
based on keyboard input. The end result of this tutorial (`mixer.c`)
will be the starter file for the assignment described below.
In this file, we create a pipeline consisting out of a videotestsrc, videoconvert and
glimagesink element.

    typedef struct _CustomData
    {
     GMainLoop* main_loop;
     GstElement* pipeline;
     GstElement* testsource;
     GstElement* convert;
     GstElement* videosink;
    } CustomData;

The first lines in the code define a struct containing all GStreamer elements (and the
GLib main loop) that will be used later. This struct allows us to easily pass
these elements to other functions or callbacks.

    #ifdef G_OS_WIN32
     io_stdin = g_io_channel_win32_new_fd(_fileno(stdin));
    #else
     io_stdin = g_io_channel_unix_new(fileno(stdin));
    #endif
     g_io_add_watch(io_stdin, G_IO_IN, (GIOFunc)handle_keyboard, &data);

The code above will create the stdin keyboard callback. This callback will be initiated
whenever a user types something in the console and ends it with ENTER (\n).

    static gboolean handle_keyboard(GIOChannel* source, GIOCondition cond, CustomData* data) {
     gchar* str = NULL;
     if (g_io_channel_read_line(source, &str, NULL, NULL, NULL) == G_IO_STATUS_NORMAL) {
      // CHECK INPUT
	  // ...
     }
     g_free(str);
     return TRUE;
    }
	
In the `handle_keyboard()` callback function, the first line of code checks if the readout
of the stdin channel was successful. If so, it will store the read string in str and return a
`G_IO_STATUS_NORMAL`.

Inside the if statement, the commands to be used in the assignment are already
defined. At this point in time, the keyboard input is only displayed in the stdout.


## Tasks
Complete the mixer.c file such that the application has the following functionality: 
1. If the user types ‘crossfade’ as a command, using the transparency (alpha) of the video streams, accomplish a gradual fade such that the other video becomes visible. One video channel (filesrc) is visualized. Allow a crossfade to a different video channel (filesrc). The crossfade command can be used indefinitely, meaning that repeatedly crossfading between the two feeds must be possible.
2. The pipeline will continue by including a video effect of your choice. When the user types ‘effect’ as a command, this effect will be replaced by another effect of your choice. There should always be an effect applied. The effect command can be used repeatedly (switching between your 2 chosen effects).
3. When the user types ‘logo’ as a command, your channel’s logo will be put in the upper left corner of the video. Typing 'logo' again should result in the logo disapearing. The logo command can be used repeatedly.
4. The output (after mixing) must be displayed on screen.
5. The output must simultaneously be stored to disk, as an H.264 MKV file. 
6. Check cross-compilation compatibility by verifying successful compilation on Ubuntu, Mac and Windows. Everytime you push your code to github, the CI runner on github will compile your code for all necessary platforms. You can verify this on your github repository. In the Actions tab, you will be able to find all compilation errors for each platform. 

## Important restrictions:

1. Use a decodebin in order to decode the videofiles, other autopluggers are not allowed.

2. Do not use gst_parse_launch, since we want you to construct the pipeline manually.

3. The path to the video files can be hard coded, but it has to be a relative path starting at the root directory of the repository.

4. Video seeking and start/stop capabilities should not be present, because we are assuming that a live stream is entering the system.

## Submission

Record a video as proof of every command working using a screengrabber, such as OBS Studio, or your smartphone. The video should be named `proof.mkv`, and be a maximum of 30 sec and 50MiB.:
* starting with your name visible 
* followed by the outputs of your shell script (playback and .mkv) as proof of a working solution. Make sure that all asked functionality can be verified. You can interrupt a working command after several seconds. 

In your submission, do not include code that does not work. If you did not implement certain functionalities, specify it clearly in the code and in the proof-video.

Before the deadline (see Ufora), **push** your code (`mixer.c`) and proof (`proof.mkv`) to the provided private repository on GitHub. Additionally, fill in the assignement **survey** on Ufora.

## Building the project using CMake

For this assignment, a `CMakeLists.txt` file is included in the repository. This takes care of finding where GStreamer is installed on your machine and correctly linking it to your project files in `/src`. For those not familiar with `CMake`, the following sections give a brief overview of how to run it and start coding. We assume that both the runtime and development GStreamer packages are already installed.

If you run into trouble or have questions during this process, do not hesitate to contact us at glenn.vanwallendael@ugent.be with in CC julie.artois@ugent.be and hannes.mareen@ugent.be .

### Linux
Install CMake (and GCC if this was not the case already, since you will need a C++ compiler). Open a terminal and navigate to the directory containing the `CMakeLists.txt`. Now run:

```
mkdir build
cd build
cmake ..
make
```
If everything goes well, you well have an executable file (might require `sudo chmod +x <filename>`). If you make changes to the files in `/src`, simply re-run `make` to rebuild the executable.

### MacOS
See the previous section. You can use the C++ compiler in XCode for example.

### Windows
Install Visual Studio (!= Visual Studio Code, which does not have a C++ compiler by default). If you are unsure whether or not your installation is sufficient, open the `Visual Studio Installer`, click "Modify" and make sure the checkbox of "Desktop development with C++" is checked.

**The recommended approach**

Open `Visual Studio`. On the right side, click "Continue without code". In the top-left corner, choose "File > Open > CMake.." and in the file explorer, select the `CMakeLists.txt` file of the assignment. This will setup your project in Visual studio. After waiting a bit, you should see a terminal opening at the bottom, running CMake. Check if there are errors (if yes, move on the workaround described below). If not, you can find the `/src` folder in Visual Studio's "Solution Explorer" on the left. Open it and click on `mixer.c` to start coding.

Make sure that at the top of Visual Studio (under "Build, Debug, Test, Analyze, etc."), the configuration is set to Debug or Release and x64. 

To build and run the project, click "Build > Build Solution" or press F7. If there are build errors, they will appear at the bottom of Visual Studio. If not, then the executable file `VideoMixer.exe` should have been created somewhere in a subdirectory. You can double click it to run it directly, or you can open a terminal, navigate to the correct folder and run the .exe file from there.

**A workaround in case CMake cannot find GStreamer**

This section is for those who get a CMake error related to:

```
CMake Error at .../PkgConfig.cmake:696 (message): None of the required 'gstreamer-app-1.0>=1.4' found
```

Firstly, did you add the GStreamer path to your environment variables (omgevingsvariabelen) like the [installation documentation specified](https://gstreamer.freedesktop.org/documentation/installing/on-windows.html?gi-language=c) specified in this paragraph?

```
How to do this: Windows start icon > Search "environment variables" > Edit the system environment variables (will open System Properties)

Environment Variables > System variables > Variable :Path > Edit > New > Paste "C:\gstreamer\1.0\msvc_x86_64\bin" > OK
```

Try to run CMake again. If the same error remains and you are in the classroom with us, please let us know. If you are at home and/or want to solve it yourself, close Visual Studio for now and follow the instructions below:

1. Install [CMake-GUI](https://cmake.org/download/) (e.g. `cmake-3.26.0-rc3-windows-x86_64.msi`)
2. Open `CmakeLists.txt`, delete all text and replace it by the following, and save of course:
```
cmake_minimum_required(VERSION 3.15.3)
project(VideoMixer)
set(CMAKE_CXX_STANDARD 20)
add_executable(VideoMixer src/mixer.c)
```
3. Open the newly installed CMake (cmake-gui) application. Right of "Where is the source code", fill in the path to the folder containing the CMakeLists.txt file. Right of "Where to build the binaries", copy past the same path, but now with "/build" at the end.
4. Press the Configure button. In the new window, click Yes if necessary, and choose your current Visual Studio version as generator and click OK. Wait until the configuring at the bottom is done and click "Generate". This should create a Visual Studio solution `VideoMixer.sln` in the `/build` folder. Do **not** open this yet.
5. Create a new text file called `VideoMixer.vcxproj.user` in the same folder as `VideoMixer.sln` and fill it with the following text:
```
<?xml version="1.0" encoding="utf-8"?>
<Project xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <Import Project="C:/gstreamer/1.0/msvc_x86_64/share/vs/2010/libs/gstreamer-1.0.props" />
  <Import Project="C:/gstreamer/1.0/msvc_x86_64/share/vs/2010/libs/gstreamer-controller-1.0.props" />
</Project>
```
Important: check if these paths to files `gstreamer-1.0.props` and `gstreamer-controller-1.0.props` are correct for your system. It might be that your GStreamer installation is in a different directory. If so, you can adapt the paths in the `VideoMixer.vcxproj.user` file.

6. Double click `VideoMixer.sln` to open it in Visual Studio.
7. Continue with the instructions from before this unfortunate and rather lengthy workaround.


## Copyright

It is crucial that copyright is never infringed. Do not base your solution on anything else than the [GStreamer documentation](https://gstreamer.freedesktop.org/documentation/index.html?gi-language=c). 
