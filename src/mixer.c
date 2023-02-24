/* ****************************************
   Name: <firstname, name>
   OS used: <Windows|OSX|Debian>
   **************************************** */
#include <gst/gst.h>
#include <stdio.h>

typedef struct _CustomData {
  GMainLoop *main_loop;
  GstElement *pipeline;
  GstElement *source_1;
  GstElement *source_2;
  GstElement *demux_1;
  GstElement *demux_2;
  GstElement *video_queue_1;
  GstElement *video_queue_2;
  GstElement *video_decodebin_1;
  GstElement *video_decodebin_2;
  GstElement *audio_decodebin_1;
  GstElement *audio_decodebin_2;
  GstElement *videoconvert_1;
  GstElement *videoconvert_2;
  GstElement *alpha_1;
  GstElement *alpha_2;
  GstElement *compositor;
  GstPad *compositor_sink_0;
  GstPad *compositor_sink_1;
  gboolean *current_video;
  GstElement *videosink;
} CustomData;

static gboolean handle_message(GstBus *bus, GstMessage *msg, CustomData *data);
static gboolean handle_keyboard(GIOChannel *source, GIOCondition cond,
                                CustomData *data);

/*declarations*/
static void crossfade(CustomData *data);
static void pad_added_handler (GstElement *src, GstPad *pad, CustomData *data);

int main(int argc, char *argv[]) {
  CustomData data;
  GstBus *bus;
  GstStateChangeReturn ret;

  GIOChannel *io_stdin;

  gst_init(&argc, &argv);

  /*source_1 up to composer*/
  data.source_1 = gst_element_factory_make("filesrc", "source_1");
  data.demux_1 = gst_element_factory_make("qtdemux", "demux_1");
  data.video_decodebin_1 = gst_element_factory_make("decodebin", "video_decodebin_1");
  data.videoconvert_1 = gst_element_factory_make("videoconvert", "videoconvert_1");
  data.alpha_1 = gst_element_factory_make("alpha", "alpha_1");

  /*source_2 up to composer*/
  data.source_2 = gst_element_factory_make("filesrc", "source_2");
  data.demux_2 = gst_element_factory_make("qtdemux", "demux_2");
  data.video_decodebin_2 = gst_element_factory_make("decodebin", "video_decodebin_2");
  data.videoconvert_2 = gst_element_factory_make("videoconvert", "videoconvert_2");
  data.alpha_2 = gst_element_factory_make("alpha", "alpha_2");

  /*composer up to videosink*/
  data.compositor = gst_element_factory_make("compositor", "compositor");
  data.videosink = gst_element_factory_make("glimagesink", "videosink");

  /*pipeline*/
  data.pipeline = gst_pipeline_new("test-pipeline");

  /*check if all elements created*/
  if (!data.pipeline || !data.source_1 || !data.demux_1 || !data.video_decodebin_1 || !data.videoconvert_1 || !data.alpha_1 || 
  !data.source_2 || !data.demux_2 || !data.video_decodebin_2 || !data.videoconvert_2 || !data.alpha_2 ||
  !data.compositor || !data.videosink) {
    g_printerr("!!! Not all elements could be created.\n");
    return -1;
  }

  /*add all elemets to bin (so they can be linked)*/
  gst_bin_add_many(GST_BIN(data.pipeline), data.source_1, data.demux_1, data.video_decodebin_1, data.videoconvert_1, data.alpha_1,
  data.source_2, data.demux_2, data.video_decodebin_2, data.videoconvert_2, data.alpha_2,
  data.compositor, data.videosink,
  NULL);
  
  /*link source_1 and demux_1*/
  if (!gst_element_link_many(data.source_1, data.demux_1,
                             NULL)) {
    g_printerr("Linking error: Unable to statically link source_1 to demux_1)\n");
    gst_object_unref(data.pipeline);
    return -1;
  }

  /*link source_2 and demux_2*/
  if (!gst_element_link_many(data.source_2, data.demux_2,
                             NULL)) {
    g_printerr("Linking error: Unable to statically link source_2 to demux_2)\n");
    gst_object_unref(data.pipeline);
    return -1;
  }

  /*link videoconvert_1 and alpha_1 and composer*/
  if (!gst_element_link_many(data.videoconvert_1, data.alpha_1, data.compositor,
                             NULL)) {
    g_printerr("Linking error: Unable to statically link videoconvert_1 to alpha_1 to composer).\n");
    gst_object_unref(data.pipeline);
    return -1;
  }

  /*link videoconvert_2 and alpha_2 and composer*/
  if (!gst_element_link_many(data.videoconvert_2, data.alpha_2, data.compositor,
                             NULL)) {
    g_printerr("Linking error: Unable to statically link videoconvert_2 to alpha_2 to composer).\n");
    gst_object_unref(data.pipeline);
    return -1;
  }

  /*link composer and videosink*/
  if (!gst_element_link_many(data.compositor, data.videosink,
                             NULL)) {
    g_printerr("Linking error: Unable to statically link composer to videosink).\n");
    gst_object_unref(data.pipeline);
    return -1;
  }

  /*set source locations*/
  g_object_set (data.source_1, "location", "../data/Spring.mp4", NULL);
  g_object_set (data.source_2, "location", "../data/Sprite.mp4", NULL);

  /*set initial background and alpha values for compositor, and register pads*/
  g_object_set (data.compositor, "background", 3, NULL);
  data.compositor_sink_0 =  gst_element_get_static_pad (data.compositor, "sink_0");
  data.compositor_sink_1 =  gst_element_get_static_pad (data.compositor, "sink_1");
  

  /* Connect to the pad-added signal for demux*/
  g_signal_connect (data.demux_1, "pad-added", G_CALLBACK (pad_added_handler), &data);
  g_signal_connect (data.video_decodebin_1, "pad-added", G_CALLBACK (pad_added_handler), &data);
  g_signal_connect (data.demux_2, "pad-added", G_CALLBACK (pad_added_handler), &data);
  g_signal_connect (data.video_decodebin_2, "pad-added", G_CALLBACK (pad_added_handler), &data);

  bus = gst_element_get_bus(data.pipeline);
  gst_bus_add_watch(bus, (GstBusFunc)handle_message, &data);

#ifdef G_OS_WIN32
  io_stdin = g_io_channel_win32_new_fd(_fileno(stdin));
#else
  io_stdin = g_io_channel_unix_new(fileno(stdin));
#endif
  g_io_add_watch(io_stdin, G_IO_IN, (GIOFunc)handle_keyboard, &data);

  ret = gst_element_set_state(data.pipeline, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    g_printerr("Unable to set the pipeline to the playing state.\n");
    gst_object_unref(data.pipeline);
    return -1;
  }

  data.main_loop = g_main_loop_new(NULL, FALSE);
  g_main_loop_run(data.main_loop);

  g_main_loop_unref(data.main_loop);
  g_io_channel_unref(io_stdin);
  gst_object_unref(bus);
  gst_element_set_state(data.pipeline, GST_STATE_NULL);
  gst_object_unref(data.pipeline);
  return 0;
}

static gboolean handle_message(GstBus *bus, GstMessage *msg, CustomData *data) {
  GError *err;
  gchar *debug_info;

  switch (GST_MESSAGE_TYPE(msg)) {
  case GST_MESSAGE_ERROR:
    gst_message_parse_error(msg, &err, &debug_info);
    g_printerr("!!! Error received from element %s: %s\n",
               GST_OBJECT_NAME(msg->src), err->message);
    g_printerr("!!! Debugging information: %s\n",
               debug_info ? debug_info : "none");
    g_clear_error(&err);
    g_free(debug_info);
    g_main_loop_quit(data->main_loop);
    break;
  case GST_MESSAGE_EOS:
    g_print("> End-Of-Stream reached.\n");
    g_main_loop_quit(data->main_loop);
    break;
  case GST_MESSAGE_STATE_CHANGED: {
    if (GST_MESSAGE_SRC(msg) == GST_OBJECT(data->pipeline)) {
      GstState old_state, new_state, pending_state;
      gst_message_parse_state_changed(msg, &old_state, &new_state,
                                      &pending_state);
      g_print("> Pipeline state changed from %s to %s.\n",
              gst_element_state_get_name(old_state),
              gst_element_state_get_name(new_state));
    }

  } break;
  }

  return TRUE;
}

static gboolean handle_keyboard(GIOChannel *source, GIOCondition cond,
                                CustomData *data) {
  gchar *str = NULL;

  if (g_io_channel_read_line(source, &str, NULL, NULL, NULL) ==
      G_IO_STATUS_NORMAL) {

    g_print("COMMAND ENTERED: %s\n", str);

    //**********************************************
    if (g_str_has_prefix(str, "crossfade")) {
      g_print("INPUT: CROSSFADE.\n");
      // call crossfade
      crossfade(data);
    } else if (g_str_has_prefix(str, "logo")) {
      g_print("INPUT: LOGO.\n");
      // call toggle_logo
    } else if (g_str_has_prefix(str, "effect")) {
      g_print("INPUT: EFFECT.\n");
      // call swap_effects
    }
    //**********************************************
    else {
      g_print("!!! INPUT NOT RECOGNIZED.\n");
    }
  }
  g_free(str);
  return TRUE;
}

static void pad_added_handler (GstElement *src, GstPad *new_pad, CustomData *data) {
  GstPad *sink_pad = NULL;
  GstPadLinkReturn ret;
  GstCaps *new_pad_caps = NULL;
  GstStructure *new_pad_struct = NULL;
  const gchar *new_pad_type = NULL;
  const gchar *new_pad_name = NULL;
  const gchar *source_name = NULL;

  g_print ("Received new pad '%s' from '%s':\n", GST_PAD_NAME (new_pad), GST_ELEMENT_NAME (src));

  /* Check the new pad's type */
  source_name = GST_ELEMENT_NAME (src);
  new_pad_name = GST_PAD_NAME (new_pad);
  new_pad_caps = gst_pad_get_current_caps (new_pad);
  new_pad_struct = gst_caps_get_structure (new_pad_caps, 0);
  new_pad_type = gst_structure_get_name (new_pad_struct);

  if (g_str_equal (source_name, "demux_1") && (g_str_has_prefix (new_pad_type, "video"))) {
      sink_pad = gst_element_get_static_pad (data->video_decodebin_1, "sink");
  } else if (g_str_equal (source_name, "video_decodebin_1")) {
    sink_pad = gst_element_get_static_pad (data->videoconvert_1, "sink");
  } else if (g_str_equal (source_name, "demux_2") && (g_str_has_prefix (new_pad_type, "video"))) {
    sink_pad = gst_element_get_static_pad (data->video_decodebin_2, "sink");
  } else if (g_str_equal (source_name, "video_decodebin_2")) {
    sink_pad = gst_element_get_static_pad (data->videoconvert_2, "sink");
  } else {
    g_print ("It has type '%s' which is not supported. Ignoring.\n", new_pad_type);
    goto exit;
  }

  /* If our converter is already linked, we have nothing to do here */
  if (gst_pad_is_linked (sink_pad)) {
    g_print ("We are already linked. Ignoring.\n");
    goto exit;
  }

  /* Attempt the link */
  ret = gst_pad_link (new_pad, sink_pad);
  if (GST_PAD_LINK_FAILED (ret)) {
    g_print ("Type is '%s' but link failed.\n", new_pad_type);
  } else {
    g_print ("Link succeeded (type '%s').\n", new_pad_type);
  }

  exit:
  /* Unreference the new pad's caps, if we got them */
  if (new_pad_caps != NULL)
    gst_caps_unref (new_pad_caps);

  /* Unreference the sink pad */
  if (sink_pad != NULL)
    gst_object_unref (sink_pad);
}

static void crossfade(CustomData *data){
  g_object_set (data->compositor_sink_0, "alpha", 0.5, NULL);
  g_object_set (data->compositor_sink_1, "alpha", 0.5, NULL);
}