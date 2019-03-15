#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/jit.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/argparser.h>
#include <mitsuba/core/xml.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/common.h>
#include <mitsuba/ui/viewer.h>

#include <tbb/tbb.h>

#include <enoki/transform.h>

using namespace mitsuba;

static void help(int thread_count) {
    std::cout << util::info_build(thread_count) << std::endl;
    std::cout << util::info_copyright() << std::endl;
    std::cout << util::info_features() << std::endl;
    std::cout << R"(
Usage: mtsgui [options] <One or more scene XML files>

Options:

   -h, --help
               Display this help text.

   -v, --verbose
               Be more verbose. (can be specified multiple times)

   -t <count>, --threads <count>
               Render with the specified number of threads.

   -D <key>=<value>, --define <key>=<value>
               Define a constant that can referenced as "$key"
               within the scene description.

   -c <configuration>, --configuration <configuration>
               Render using the specified configuration of Mitsuba.
               The following are currently available:

                   'scalar-mono'
                   'scalar-rgb'
                   'scalar-spectral'
                   'scalar-spectral-polarized'
                   'packet-mono'
                   'packet-rgb'
                   'packet-spectral'
                   'packet-spectral-polarized'
                   'gpu-autodiff-mono'
                   'gpu-autodiff-rgb'
                   'gpu-autodiff-spectral'
                   'gpu-autodiff-spectral-polarized'

               The default is 'scalar-spectral'.
)";
}

int main(int argc, char *argv[]) {
    Jit::static_initialization();
    Class::static_initialization();
    Thread::static_initialization();
    Logger::static_initialization();
    Bitmap::static_initialization();

    /* Ensure that the mitsuba-render shared library is loaded */
    librender_nop();

    ArgParser parser;
    using StringVec = std::vector<std::string>;
    auto arg_threads = parser.add(StringVec { "-t", "--threads" }, true);
    auto arg_verbose = parser.add(StringVec { "-v", "--verbose" }, false);
    auto arg_define  = parser.add(StringVec { "-D", "--define" }, true);
    auto arg_config = parser.add(StringVec{ "-c", "--configuration" }, false);
    auto arg_help = parser.add(StringVec { "-h", "--help" });
    auto arg_extra = parser.add("", true);
    xml::ParameterList params;

    std::string error_msg;
    try {
        // Parse all command line options
        parser.parse(argc, argv);

        if (*arg_verbose) {
            auto logger = Thread::thread()->logger();
            if (arg_verbose->next())
                logger->set_log_level(ETrace);
            else
                logger->set_log_level(EDebug);
        }

        while (arg_define && *arg_define) {
            std::string value = arg_define->as_string();
            auto sep = value.find('=');
            if (sep == std::string::npos)
                Throw("-D/--define: expect key=value pair!");
            params.push_back(std::make_pair(value.substr(0, sep),
                                            value.substr(sep+1)));
            arg_define = arg_define->next();
        }

        // Initialize Intel Thread Building Blocks with the requested number of threads
        if (*arg_threads)
            __global_thread_count = arg_threads->as_int();
        if (__global_thread_count < 1)
            Throw("Thread count must be >= 1!");
        tbb::task_scheduler_init init((int) __global_thread_count);

        // Append the mitsuba directory to the FileResolver search path list
        ref<Thread> thread = Thread::thread();
        ref<FileResolver> fr = thread->file_resolver();
        filesystem::path base_path = util::library_path().parent_path();
        if (!fr->contains(base_path))
            fr->append(base_path);

        if (*arg_help) {
            help((int) __global_thread_count);
        } else {
            ng::init();

            {
                ng::ref<MitsubaViewer> viewer = new MitsubaViewer();
                viewer->dec_ref();

                // Initialize profiler *after* NanoGUI
                Profiler::static_initialization();

                ThreadEnvironment env;
                tbb::task_group group;

                while (arg_extra && *arg_extra) {
                    filesystem::path filename(arg_extra->as_string());

                    MitsubaViewer::Tab *tab = viewer->append_tab(filename.filename().string());

                    group.run(
                        [&env, fr, filename, tab, viewer]() {
                            ScopedSetThreadEnvironment set_env(env);

                            // Add the scene file's directory to the search path.
                            fs::path scene_dir = filename.parent_path();
                            ref<FileResolver> fr2 = new FileResolver(*fr);
                            if (!fr2->contains(scene_dir))
                                fr2->append(scene_dir);

                            Thread *thread = Thread::thread();
                            thread->set_file_resolver(fr2);
                            ((MitsubaViewer *) viewer.get())->load(tab, fr2->resolve(filename));
                        }
                    );

                    arg_extra = arg_extra->next();
                }

                viewer->draw_all();
                viewer->set_visible(true);
                ng::mainloop(-1);

                group.wait();
            }

            ng::shutdown();
        }
    } catch (const std::exception &e) {
        error_msg = std::string("Caught a critical exception: ") + e.what();
    } catch (...) {
        error_msg = std::string("Caught a critical exception of unknown type!");
    }

    if (!error_msg.empty()) {
        /* Strip zero-width spaces from the message (Mitsuba uses these
           to properly format chains of multiple exceptions) */
        const std::string zerowidth_space = "\xe2\x80\x8b";
        while (true) {
            auto it = error_msg.find(zerowidth_space);
            if (it == std::string::npos)
                break;
            error_msg = error_msg.substr(0, it) + error_msg.substr(it + 3);
        }

#if defined(__WINDOWS__)
        HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO console_info;
        GetConsoleScreenBufferInfo(console, &console_info);
        SetConsoleTextAttribute(console, FOREGROUND_RED | FOREGROUND_INTENSITY);
#else
        std::cerr << "\x1b[31m";
#endif
        std::cerr << std::endl << error_msg << std::endl;
#if defined(__WINDOWS__)
        SetConsoleTextAttribute(console, console_info.wAttributes);
#else
        std::cerr << "\x1b[0m";
#endif
    }

    Profiler::static_shutdown();
    Bitmap::static_shutdown();
    Logger::static_shutdown();
    Thread::static_shutdown();
    Class::static_shutdown();
    Jit::static_shutdown();

    return 0;
}