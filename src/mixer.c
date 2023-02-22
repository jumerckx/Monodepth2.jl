/* ****************************************
   Name: <firstname, name>
   OS used: <Windows|OSX|Debian>
   **************************************** */
#include <gst/gst.h>
#include <stdio.h>

typedef struct _CustomData {
  GMainLoop *main_loop;
  GstElement *pipeline;
  GstElement *testsource;
  GstElement *convert;
  GstElement *videosink;
} CustomData;

static gboolean handle_message(GstBus *bus, GstMessage *msg, CustomData *data);
static gboolean handle_keyboard(GIOChannel *source, GIOCondition cond,
                                CustomData *data);

int main(int argc, char *argv[]) {
  CustomData data;
  GstBus *bus;
  GstStateChangeReturn ret;

  GIOChannel *io_stdin;

  gst_init(&argc, &argv);

  data.testsource = gst_element_factory_make("videotestsrc", "testsource");
  data.convert = gst_element_factory_make("videoconvert", "convert");
  data.videosink = gst_element_factory_make("glimagesink", "videosink");

  data.pipeline = gst_pipeline_new("test-pipeline");

  if (!data.pipeline || !data.testsource || !data.convert || !data.videosink) {
    g_printerr("!!! Not all elements could be created.\n");
    return -1;
  }

  gst_bin_add_many(GST_BIN(data.pipeline), data.testsource, data.convert,
                   data.videosink, NULL);

  if (!gst_element_link_many(data.testsource, data.convert, data.videosink,
                             NULL)) {
    g_printerr("Linking error: Unable to statically link (source to (convert) "
               "to videosink).\n");
    gst_object_unref(data.pipeline);
    return -1;
  }

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
