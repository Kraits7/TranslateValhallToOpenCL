
#include <jni.h>
#include <dlfcn.h>
#include <CL/cl.h>
#include <android/log.h>
#include <vector>
#include <string>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#define TAG "HOT30i_Compute_Core"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

typedef cl_int (CL_API_CALL *pfn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_API_CALL *pfn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (CL_API_CALL *pfn_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void (CL_CALLBACK *)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (CL_API_CALL *pfn_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_program (CL_API_CALL *pfn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (CL_API_CALL *pfn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (CL_CALLBACK *)(cl_program, void*), void*);
typedef cl_kernel (CL_API_CALL *pfn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (CL_API_CALL *pfn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (CL_API_CALL *pfn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (CL_API_CALL *pfn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL *pfn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL *pfn_clFinish)(cl_command_queue);
typedef cl_int (CL_API_CALL *pfn_clReleaseMemObject)(cl_mem);
typedef cl_int (CL_API_CALL *pfn_clReleaseKernel)(cl_kernel);
typedef cl_int (CL_API_CALL *pfn_clReleaseProgram)(cl_program);
typedef cl_int (CL_API_CALL *pfn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (CL_API_CALL *pfn_clReleaseContext)(cl_context);

class OpenCLManager {
public:
    void* handle = nullptr;
    pfn_clGetPlatformIDs clGetPlatformIDs = nullptr;
    pfn_clGetDeviceIDs clGetDeviceIDs = nullptr;
    pfn_clCreateContext clCreateContext = nullptr;
    pfn_clCreateCommandQueue clCreateCommandQueue = nullptr;
    pfn_clCreateProgramWithSource clCreateProgramWithSource = nullptr;
    pfn_clBuildProgram clBuildProgram = nullptr;
    pfn_clCreateKernel clCreateKernel = nullptr;
    pfn_clCreateBuffer clCreateBuffer = nullptr;
    pfn_clSetKernelArg clSetKernelArg = nullptr;
    pfn_clEnqueueNDRangeKernel clEnqueueNDRangeKernel = nullptr;
    pfn_clEnqueueReadBuffer clEnqueueReadBuffer = nullptr;
    pfn_clFinish clFinish = nullptr;
    pfn_clReleaseMemObject clReleaseMemObject = nullptr;
    pfn_clReleaseKernel clReleaseKernel = nullptr;
    pfn_clReleaseProgram clReleaseProgram = nullptr;
    pfn_clReleaseCommandQueue clReleaseCommandQueue = nullptr;
    pfn_clReleaseContext clReleaseContext = nullptr;

    bool init() {
        handle = dlopen("libOpenCL.so", RTLD_NOW);
        if (!handle) {
            LOGE("Critical: libOpenCL.so not found in system namespaces.");
            return false;
        }

        #define BIND_FUNC(name) name = (pfn_##name)dlsym(handle, #name)
        BIND_FUNC(clGetPlatformIDs); BIND_FUNC(clGetDeviceIDs); BIND_FUNC(clCreateContext);
        BIND_FUNC(clCreateCommandQueue); BIND_FUNC(clCreateProgramWithSource); BIND_FUNC(clBuildProgram);
        BIND_FUNC(clCreateKernel); BIND_FUNC(clCreateBuffer); BIND_FUNC(clSetKernelArg);
        BIND_FUNC(clEnqueueNDRangeKernel); BIND_FUNC(clEnqueueReadBuffer); BIND_FUNC(clFinish);
        BIND_FUNC(clReleaseMemObject); BIND_FUNC(clReleaseKernel); BIND_FUNC(clReleaseProgram);
        BIND_FUNC(clReleaseCommandQueue); BIND_FUNC(clReleaseContext);

        LOGI("OpenCL symbols bound successfully from system vendor library.");
        return clGetPlatformIDs != nullptr;
    }

    ~OpenCLManager() { if (handle) dlclose(handle); }
};

static OpenCLManager ocl;

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_compute_GpuEngine_processData(JNIEnv* env, jobject thiz, jfloatArray input) {
    if (!ocl.handle && !ocl.init()) return input;

    jsize len = env->GetArrayLength(input);
    if (len == 0) return input;

    jfloat* ptr = env->GetFloatArrayElements(input, nullptr);

    cl_int err;
    cl_platform_id platform;
    ocl.clGetPlatformIDs(1, &platform, nullptr);

    cl_device_id device;
    ocl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context ctx = ocl.clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = ocl.clCreateCommandQueue(ctx, device, 0, &err);

    const char* kernelSrc = R"(
        __kernel void mali_turbo_kernel(__global float4* in, __global float4* out) {
            int id = get_global_id(0);
            float4 val = in[id];
            out[id] = native_sqrt(val * val + 1.0f);
        }
    )";

    cl_program prog = ocl.clCreateProgramWithSource(ctx, 1, &kernelSrc, nullptr, &err);
    ocl.clBuildProgram(prog, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
    cl_kernel kernel = ocl.clCreateKernel(prog, "mali_turbo_kernel", &err);

    cl_mem bIn = ocl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * len, ptr, &err);
    cl_mem bOut = ocl.clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * len, nullptr, &err);

    ocl.clSetKernelArg(kernel, 0, sizeof(cl_mem), &bIn);
    ocl.clSetKernelArg(kernel, 1, sizeof(cl_mem), &bOut);

    size_t global_size = len / 4; 
    ocl.clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    ocl.clFinish(queue);

    ocl.clEnqueueReadBuffer(queue, bOut, CL_TRUE, 0, sizeof(float) * len, ptr, 0, nullptr, nullptr);

    ocl.clReleaseMemObject(bIn); ocl.clReleaseMemObject(bOut);
    ocl.clReleaseKernel(kernel); ocl.clReleaseProgram(prog);
    ocl.clReleaseCommandQueue(queue); ocl.clReleaseContext(ctx);

    env->ReleaseFloatArrayElements(input, ptr, 0);
    return input;
}
