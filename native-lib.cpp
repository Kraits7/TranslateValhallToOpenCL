#include <jni.h>
#include <dlfcn.h>
#include <CL/cl.h>
#include <android/log.h>
#include <vector>
#include <string>

#define TAG "HOT30i_G57_Engine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

typedef cl_int (*TYPE_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*TYPE_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*TYPE_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void (*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (*TYPE_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_program (*TYPE_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*TYPE_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*), void*);
typedef cl_kernel (*TYPE_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (*TYPE_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (*TYPE_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*TYPE_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*TYPE_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*TYPE_clFinish)(cl_command_queue);

class OpenCLDriver {
public:
    void* libHandle = nullptr;
    TYPE_clGetPlatformIDs clGetPlatformIDs;
    TYPE_clGetDeviceIDs clGetDeviceIDs;
    TYPE_clCreateContext clCreateContext;
    TYPE_clCreateCommandQueue clCreateCommandQueue;
    TYPE_clCreateProgramWithSource clCreateProgramWithSource;
    TYPE_clBuildProgram clBuildProgram;
    TYPE_clCreateKernel clCreateKernel;
    TYPE_clCreateBuffer clCreateBuffer;
    TYPE_clSetKernelArg clSetKernelArg;
    TYPE_clEnqueueNDRangeKernel clEnqueueNDRangeKernel;
    TYPE_clEnqueueReadBuffer clEnqueueReadBuffer;
    TYPE_clFinish clFinish;

    bool load() {
        libHandle = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW);
        if (!libHandle) {
            LOGE("Не удалось загрузить библиотеку: %s", dlerror());
            return false;
        }

        clGetPlatformIDs = (TYPE_clGetPlatformIDs)dlsym(libHandle, "clGetPlatformIDs");
        clGetDeviceIDs = (TYPE_clGetDeviceIDs)dlsym(libHandle, "clGetDeviceIDs");
        clCreateContext = (TYPE_clCreateContext)dlsym(libHandle, "clCreateContext");
        clCreateCommandQueue = (TYPE_clCreateCommandQueue)dlsym(libHandle, "clCreateCommandQueue");
        clCreateProgramWithSource = (TYPE_clCreateProgramWithSource)dlsym(libHandle, "clCreateProgramWithSource");
        clBuildProgram = (TYPE_clBuildProgram)dlsym(libHandle, "clBuildProgram");
        clCreateKernel = (TYPE_clCreateKernel)dlsym(libHandle, "clCreateKernel");
        clCreateBuffer = (TYPE_clCreateBuffer)dlsym(libHandle, "clCreateBuffer");
        clSetKernelArg = (TYPE_clSetKernelArg)dlsym(libHandle, "clSetKernelArg");
        clEnqueueNDRangeKernel = (TYPE_clEnqueueNDRangeKernel)dlsym(libHandle, "clEnqueueNDRangeKernel");
        clEnqueueReadBuffer = (TYPE_clEnqueueReadBuffer)dlsym(libHandle, "clEnqueueReadBuffer");
        clFinish = (TYPE_clFinish)dlsym(libHandle, "clFinish");

        return clGetPlatformIDs != nullptr;
    }

    ~OpenCLDriver() {
        if (libHandle) dlclose(libHandle);
    }
};

static OpenCLDriver driver;

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_compute_GpuEngine_process(JNIEnv* env, jobject thiz, jfloatArray input_data) {
    if (!driver.libHandle && !driver.load()) {
        LOGE("Драйвер OpenCL недоступен.");
        return input_data;
    }

    jsize len = env->GetArrayLength(input_data);
    jfloat* data_ptr = env->GetFloatArrayElements(input_data, nullptr);

    cl_int err;
    cl_platform_id platform;
    driver.clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    driver.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = driver.clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = driver.clCreateCommandQueue(context, device, 0, &err);

    const char* src = 
        "__kernel void fast_process_v1(__global float4* in, __global float4* out) {"
        "    int id = get_global_id(0);"
        "    out[id] = native_sqrt(in[id] * in[id] + 1.0f);"
        "}";

    cl_program program = driver.clCreateProgramWithSource(context, 1, &src, nullptr, &err);
    driver.clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = driver.clCreateKernel(program, "fast_process_v1", &err);

    cl_mem bufIn = driver.clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * len, data_ptr, &err);
    cl_mem bufOut = driver.clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * len, nullptr, &err);

    driver.clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufIn);
    driver.clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufOut);

    size_t global_size = len / 4;
    driver.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    
    driver.clFinish(queue);

    driver.clEnqueueReadBuffer(queue, bufOut, CL_TRUE, 0, sizeof(float) * len, data_ptr, 0, nullptr, nullptr);
  
    env->ReleaseFloatArrayElements(input_data, data_ptr, 0);
    
    return input_data;
}
