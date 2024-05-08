GCC = gcc 
NVCC = nvcc 
OPTIONS1 = -c -g
OPTIONS2 = -lm 
OPTIONS4 = -L/apps/easybuild/software/infer-skylake/CUDAcore/11.1.1/bin/nvcc -arch=sm_61 -G


OBJECTS =   \
  matself.o			\



all:		\
  .o  	\
  parallel.o       	\
  link



r_gmres_cd.o		: r_gmres_cd.c
		$(GCC) -I. $(OPTIONS1) r_gmres_cd.c

daxpy.o			: daxpy.c
		$(GCC) $(OPTIONS1) daxpy.c

daxpyCU.o	: daxpyCU.cu
		$(NVCC) $(OPTIONS5) $(OPTIONS1) daxpyCU.cu $(OPTIONS4) -lcuda -lcudart -lcublas

givens.o		: givens.c
		$(GCC) $(OPTIONS1) givens.c

prevgiv.o		: prevgiv.c
		$(GCC) $(OPTIONS1) prevgiv.c

uptrisol.o		: uptrisol.c
		$(GCC) $(OPTIONS1) uptrisol.c

matvec.o		: matvec.c
		$(GCC) $(OPTIONS1) matvec.c

matvec_CSR.o	: matvec_CSR.cu
		$(NVCC) $(OPTIONS1) matvec_CSR.cu $(OPTIONS4) -lcuda -lcudart -lcublas

dscal.o			: dscal.c
		$(GCC) $(OPTIONS1) dscal.c

dscalCU.o	: dscalCU.cu
		$(NVCC) $(OPTIONS5) $(OPTIONS1) dscalCU.cu $(OPTIONS4) -lcuda -lcudart -lcublas

dzero.o			: dzero.c
		$(GCC) $(OPTIONS1) dzero.c

ddot.o			: ddot.c
		$(GCC) $(OPTIONS1) ddot.c

ddotCU.o	: ddotCU.cu
		$(NVCC) $(OPTIONS5) $(OPTIONS1) ddotCU.cu $(OPTIONS4) -lcuda -lcudart -lcublas

dnrm2.o			: dnrm2.c
		$(GCC) $(OPTIONS1) dnrm2.c

dnrm2CU.o	: dnrm2CU.cu
		$(NVCC) $(OPTIONS5) $(OPTIONS1) dnrm2CU.cu $(OPTIONS4) -lcuda -lcudart -lcublas

dcopy.o			: dcopy.c
		$(GCC) $(OPTIONS1) dcopy.c

dcopyCU.o	: dcopyCU.cu
		$(NVCC) $(OPTIONS5) $(OPTIONS1) dcopyCU.cu $(OPTIONS4) -lcuda -lcudart -lcublas

sln_file.o			: sln_file.c
		$(GCC) $(OPTIONS1) sln_file.c

gmresm.o		: gmresm.c
		$(GCC) -I. $(OPTIONS1) gmresm.c

link			:
		nvcc -arch=sm_70 $(OBJECTS) $(OPTIONS2) -o r_gmres_cd -lcuda -lcudart -lcublas

clean			:
		rm *.o r_gmres_cd
