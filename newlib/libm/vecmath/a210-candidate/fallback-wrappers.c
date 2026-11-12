#include <vecmath.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#define fabs __builtin_fabs
#define fabsf __builtin_fabsf
extern void a210_scalar_vsin(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vsin(double *__restrict y, const double *__restrict x, size_t n);
void vsin(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vsin(y, x, n);}
extern void a210_scalar_vcos(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vcos(double *__restrict y, const double *__restrict x, size_t n);
void vcos(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vcos(y, x, n);}
extern void a210_scalar_vtan(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vtan(double *__restrict y, const double *__restrict x, size_t n);
void vtan(double *__restrict y, const double *__restrict x, size_t n){a210_rvv_vtan(y, x, n);}
extern void a210_scalar_vsincos(double *__restrict sy, double *__restrict cy,
	      const double *__restrict x, size_t n);
extern void a210_rvv_vsincos(double *__restrict sy, double *__restrict cy,
	      const double *__restrict x, size_t n);
void vsincos(double *__restrict sy, double *__restrict cy,
	      const double *__restrict x, size_t n){a210_scalar_vsincos(sy, cy, x, n);}
extern void a210_scalar_vsinpi(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vsinpi(double *__restrict y, const double *__restrict x, size_t n);
void vsinpi(double *__restrict y, const double *__restrict x, size_t n){a210_rvv_vsinpi(y, x, n);}
extern void a210_scalar_vcospi(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vcospi(double *__restrict y, const double *__restrict x, size_t n);
void vcospi(double *__restrict y, const double *__restrict x, size_t n){a210_rvv_vcospi(y, x, n);}
extern void a210_scalar_vtanpi(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vtanpi(double *__restrict y, const double *__restrict x, size_t n);
void vtanpi(double *__restrict y, const double *__restrict x, size_t n){a210_rvv_vtanpi(y, x, n);}
extern void a210_scalar_vsincospi(double *__restrict sy, double *__restrict cy,
		const double *__restrict x, size_t n);
extern void a210_rvv_vsincospi(double *__restrict sy, double *__restrict cy,
		const double *__restrict x, size_t n);
void vsincospi(double *__restrict sy, double *__restrict cy,
		const double *__restrict x, size_t n){a210_rvv_vsincospi(sy, cy, x, n);}
extern void a210_scalar_vasin(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vasin(double *__restrict y, const double *__restrict x, size_t n);
void vasin(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vasin(y, x, n);}
extern void a210_scalar_vacos(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vacos(double *__restrict y, const double *__restrict x, size_t n);
void vacos(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vacos(y, x, n);}
extern void a210_scalar_vatan(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vatan(double *__restrict y, const double *__restrict x, size_t n);
void vatan(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vatan(y, x, n);}
extern void a210_scalar_vatan2(double *__restrict y, const double *__restrict ay,
	     const double *__restrict ax, size_t n);
extern void a210_rvv_vatan2(double *__restrict y, const double *__restrict ay,
	     const double *__restrict ax, size_t n);
void vatan2(double *__restrict y, const double *__restrict ay,
	     const double *__restrict ax, size_t n){a210_scalar_vatan2(y, ay, ax, n);}
extern void a210_scalar_vasinpi(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vasinpi(double *__restrict y, const double *__restrict x, size_t n);
void vasinpi(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vasinpi(y, x, n);}
extern void a210_scalar_vacospi(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vacospi(double *__restrict y, const double *__restrict x, size_t n);
void vacospi(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vacospi(y, x, n);}
extern void a210_scalar_vatanpi(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vatanpi(double *__restrict y, const double *__restrict x, size_t n);
void vatanpi(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vatanpi(y, x, n);}
extern void a210_scalar_vatan2pi(double *__restrict y, const double *__restrict ay,
	       const double *__restrict ax, size_t n);
extern void a210_rvv_vatan2pi(double *__restrict y, const double *__restrict ay,
	       const double *__restrict ax, size_t n);
void vatan2pi(double *__restrict y, const double *__restrict ay,
	       const double *__restrict ax, size_t n){size_t start=0; for(size_t i=0;i<n;i++) if(!isfinite(ay[i]) || !isfinite(ax[i]) || ay[i] == 0.0 || ax[i] == 0.0 || fabs(ay[i]) < 0x1p-1000) {
 if(i>start) a210_scalar_vatan2pi(y+start, ay+start, ax+start,i-start);
 a210_rvv_vatan2pi(y+i, ay+i, ax+i, 1); start=i+1;
 } if(start<n) a210_scalar_vatan2pi(y+start, ay+start, ax+start,n-start);}
extern void a210_scalar_vsinh(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vsinh(double *__restrict y, const double *__restrict x, size_t n);
void vsinh(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vsinh(y, x, n);}
extern void a210_scalar_vcosh(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vcosh(double *__restrict y, const double *__restrict x, size_t n);
void vcosh(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vcosh(y, x, n);}
extern void a210_scalar_vtanh(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vtanh(double *__restrict y, const double *__restrict x, size_t n);
void vtanh(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vtanh(y, x, n);}
extern void a210_scalar_vasinh(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vasinh(double *__restrict y, const double *__restrict x, size_t n);
void vasinh(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vasinh(y, x, n);}
extern void a210_scalar_vacosh(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vacosh(double *__restrict y, const double *__restrict x, size_t n);
void vacosh(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vacosh(y, x, n);}
extern void a210_scalar_vatanh(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vatanh(double *__restrict y, const double *__restrict x, size_t n);
void vatanh(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vatanh(y, x, n);}
extern void a210_scalar_vexp(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vexp(double *__restrict y, const double *__restrict x, size_t n);
void vexp(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vexp(y, x, n);}
extern void a210_scalar_vexp2(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vexp2(double *__restrict y, const double *__restrict x, size_t n);
void vexp2(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vexp2(y, x, n);}
extern void a210_scalar_vexp10(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vexp10(double *__restrict y, const double *__restrict x, size_t n);
void vexp10(double *__restrict y, const double *__restrict x, size_t n){a210_rvv_vexp10(y, x, n);}
extern void a210_scalar_vexpm1(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vexpm1(double *__restrict y, const double *__restrict x, size_t n);
void vexpm1(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vexpm1(y, x, n);}
extern void a210_scalar_vlog(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vlog(double *__restrict y, const double *__restrict x, size_t n);
void vlog(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vlog(y, x, n);}
extern void a210_scalar_vlog2(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vlog2(double *__restrict y, const double *__restrict x, size_t n);
void vlog2(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vlog2(y, x, n);}
extern void a210_scalar_vlog10(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vlog10(double *__restrict y, const double *__restrict x, size_t n);
void vlog10(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vlog10(y, x, n);}
extern void a210_scalar_vlog1p(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vlog1p(double *__restrict y, const double *__restrict x, size_t n);
void vlog1p(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vlog1p(y, x, n);}
extern void a210_scalar_vlog2p1(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vlog2p1(double *__restrict y, const double *__restrict x, size_t n);
void vlog2p1(double *__restrict y, const double *__restrict x, size_t n){size_t start=0; for(size_t i=0;i<n;i++) if(isfinite(x[i]) && (fabs(x[i]) < 0x1p-900)) {
 if(i>start) a210_scalar_vlog2p1(y+start, x+start,i-start);
 a210_rvv_vlog2p1(y+i, x+i, 1); start=i+1;
 } if(start<n) a210_scalar_vlog2p1(y+start, x+start,n-start);}
extern void a210_scalar_vpow(double *__restrict y, const double *__restrict ax,
	   const double *__restrict bx, size_t n);
extern void a210_rvv_vpow(double *__restrict y, const double *__restrict ax,
	   const double *__restrict bx, size_t n);
void vpow(double *__restrict y, const double *__restrict ax,
	   const double *__restrict bx, size_t n){a210_scalar_vpow(y, ax, bx, n);}
extern void a210_scalar_vsqrt(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vsqrt(double *__restrict y, const double *__restrict x, size_t n);
void vsqrt(double *__restrict y, const double *__restrict x, size_t n){a210_rvv_vsqrt(y, x, n);}
extern void a210_scalar_vrsqrt(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vrsqrt(double *__restrict y, const double *__restrict x, size_t n);
void vrsqrt(double *__restrict y, const double *__restrict x, size_t n){a210_rvv_vrsqrt(y, x, n);}
extern void a210_scalar_vcbrt(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vcbrt(double *__restrict y, const double *__restrict x, size_t n);
void vcbrt(double *__restrict y, const double *__restrict x, size_t n){a210_rvv_vcbrt(y, x, n);}
extern void a210_scalar_vhypot(double *__restrict y, const double *__restrict ax,
	     const double *__restrict bx, size_t n);
extern void a210_rvv_vhypot(double *__restrict y, const double *__restrict ax,
	     const double *__restrict bx, size_t n);
void vhypot(double *__restrict y, const double *__restrict ax,
	     const double *__restrict bx, size_t n){size_t start=0; for(size_t i=0;i<n;i++) if(isfinite(bx[i]) && (isfinite(ax[i]) && (fabs(ax[i]) < 0x1p-450 || fabs(bx[i]) < 0x1p-450 || fabs(ax[i]) > 0x1p450 || fabs(bx[i]) > 0x1p450))) {
 if(i>start) a210_scalar_vhypot(y+start, ax+start, bx+start,i-start);
 a210_rvv_vhypot(y+i, ax+i, bx+i, 1); start=i+1;
 } if(start<n) a210_scalar_vhypot(y+start, ax+start, bx+start,n-start);}
extern void a210_scalar_verf(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_verf(double *__restrict y, const double *__restrict x, size_t n);
void verf(double *__restrict y, const double *__restrict x, size_t n){size_t start=0; for(size_t i=0;i<n;i++) if(isfinite(x[i]) && (fabs(x[i]) < 0x1p-1000)) {
 if(i>start) a210_scalar_verf(y+start, x+start,i-start);
 a210_rvv_verf(y+i, x+i, 1); start=i+1;
 } if(start<n) a210_scalar_verf(y+start, x+start,n-start);}
extern void a210_scalar_verfc(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_verfc(double *__restrict y, const double *__restrict x, size_t n);
void verfc(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_verfc(y, x, n);}
extern void a210_scalar_vtgamma(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vtgamma(double *__restrict y, const double *__restrict x, size_t n);
extern double newlib_scalar_tgamma(double);
void vtgamma(double *__restrict y, const double *__restrict x, size_t n){for(size_t i=0;i<n;i++) { double a=x[i];
 if(!isfinite(a) || a>32 || a < -32 || (a<=0 && a==(int)a) || fabs(a)<0x1p-100) a210_rvv_vtgamma(y+i,x+i,1);
 else y[i]=newlib_scalar_tgamma(a); }}
extern void a210_scalar_vlgamma(double *__restrict y, const double *__restrict x, size_t n);
extern void a210_rvv_vlgamma(double *__restrict y, const double *__restrict x, size_t n);
void vlgamma(double *__restrict y, const double *__restrict x, size_t n){a210_scalar_vlgamma(y, x, n);}
extern void a210_scalar_vlgamma_r(double *__restrict y, int *__restrict sign,
		const double *__restrict x, size_t n);
extern void a210_rvv_vlgamma_r(double *__restrict y, int *__restrict sign,
		const double *__restrict x, size_t n);
void vlgamma_r(double *__restrict y, int *__restrict sign,
		const double *__restrict x, size_t n){a210_rvv_vlgamma_r(y, sign, x, n);}
extern void a210_scalar_vsinf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vsinf(float *__restrict y, const float *__restrict x, size_t n);
void vsinf(float *__restrict y, const float *__restrict x, size_t n){size_t start=0; for(size_t i=0;i<n;i++) if(isfinite(x[i]) && (fabsf(x[i]) < 0x1p-120f)) {
 if(i>start) a210_scalar_vsinf(y+start, x+start,i-start);
 a210_rvv_vsinf(y+i, x+i, 1); start=i+1;
 } if(start<n) a210_scalar_vsinf(y+start, x+start,n-start);}
extern void a210_scalar_vcosf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vcosf(float *__restrict y, const float *__restrict x, size_t n);
void vcosf(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vcosf(y, x, n);}
extern void a210_scalar_vtanf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vtanf(float *__restrict y, const float *__restrict x, size_t n);
extern double newlib_scalar_tan(double);
void vtanf(float *__restrict y, const float *__restrict x, size_t n){for(size_t i=0;i<n;i++) y[i]=(float)newlib_scalar_tan((double)x[i]);}
extern void a210_scalar_vsincosf(float *__restrict sy, float *__restrict cy,
	       const float *__restrict x, size_t n);
extern void a210_rvv_vsincosf(float *__restrict sy, float *__restrict cy,
	       const float *__restrict x, size_t n);
void vsincosf(float *__restrict sy, float *__restrict cy,
	       const float *__restrict x, size_t n){a210_rvv_vsincosf(sy, cy, x, n);}
extern void a210_scalar_vsinpif(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vsinpif(float *__restrict y, const float *__restrict x, size_t n);
void vsinpif(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vsinpif(y, x, n);}
extern void a210_scalar_vcospif(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vcospif(float *__restrict y, const float *__restrict x, size_t n);
void vcospif(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vcospif(y, x, n);}
extern void a210_scalar_vtanpif(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vtanpif(float *__restrict y, const float *__restrict x, size_t n);
void vtanpif(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vtanpif(y, x, n);}
extern void a210_scalar_vsincospif(float *__restrict sy, float *__restrict cy,
		 const float *__restrict x, size_t n);
extern void a210_rvv_vsincospif(float *__restrict sy, float *__restrict cy,
		 const float *__restrict x, size_t n);
void vsincospif(float *__restrict sy, float *__restrict cy,
		 const float *__restrict x, size_t n){a210_rvv_vsincospif(sy, cy, x, n);}
extern void a210_scalar_vasinf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vasinf(float *__restrict y, const float *__restrict x, size_t n);
void vasinf(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vasinf(y, x, n);}
extern void a210_scalar_vacosf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vacosf(float *__restrict y, const float *__restrict x, size_t n);
void vacosf(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vacosf(y, x, n);}
extern void a210_scalar_vatanf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vatanf(float *__restrict y, const float *__restrict x, size_t n);
void vatanf(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vatanf(y, x, n);}
extern void a210_scalar_vatan2f(float *__restrict y, const float *__restrict ay,
	      const float *__restrict ax, size_t n);
extern void a210_rvv_vatan2f(float *__restrict y, const float *__restrict ay,
	      const float *__restrict ax, size_t n);
void vatan2f(float *__restrict y, const float *__restrict ay,
	      const float *__restrict ax, size_t n){a210_rvv_vatan2f(y, ay, ax, n);}
extern void a210_scalar_vasinpif(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vasinpif(float *__restrict y, const float *__restrict x, size_t n);
void vasinpif(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vasinpif(y, x, n);}
extern void a210_scalar_vacospif(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vacospif(float *__restrict y, const float *__restrict x, size_t n);
void vacospif(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vacospif(y, x, n);}
extern void a210_scalar_vatanpif(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vatanpif(float *__restrict y, const float *__restrict x, size_t n);
void vatanpif(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vatanpif(y, x, n);}
extern void a210_scalar_vatan2pif(float *__restrict y, const float *__restrict ay,
		const float *__restrict ax, size_t n);
extern void a210_rvv_vatan2pif(float *__restrict y, const float *__restrict ay,
		const float *__restrict ax, size_t n);
void vatan2pif(float *__restrict y, const float *__restrict ay,
		const float *__restrict ax, size_t n){a210_rvv_vatan2pif(y, ay, ax, n);}
extern void a210_scalar_vsinhf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vsinhf(float *__restrict y, const float *__restrict x, size_t n);
void vsinhf(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vsinhf(y, x, n);}
extern void a210_scalar_vcoshf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vcoshf(float *__restrict y, const float *__restrict x, size_t n);
void vcoshf(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vcoshf(y, x, n);}
extern void a210_scalar_vtanhf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vtanhf(float *__restrict y, const float *__restrict x, size_t n);
void vtanhf(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vtanhf(y, x, n);}
extern void a210_scalar_vasinhf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vasinhf(float *__restrict y, const float *__restrict x, size_t n);
void vasinhf(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vasinhf(y, x, n);}
extern void a210_scalar_vacoshf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vacoshf(float *__restrict y, const float *__restrict x, size_t n);
void vacoshf(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vacoshf(y, x, n);}
extern void a210_scalar_vatanhf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vatanhf(float *__restrict y, const float *__restrict x, size_t n);
void vatanhf(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vatanhf(y, x, n);}
extern void a210_scalar_vexpf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vexpf(float *__restrict y, const float *__restrict x, size_t n);
void vexpf(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vexpf(y, x, n);}
extern void a210_scalar_vexp2f(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vexp2f(float *__restrict y, const float *__restrict x, size_t n);
void vexp2f(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vexp2f(y, x, n);}
extern void a210_scalar_vexp10f(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vexp10f(float *__restrict y, const float *__restrict x, size_t n);
void vexp10f(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vexp10f(y, x, n);}
extern void a210_scalar_vexpm1f(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vexpm1f(float *__restrict y, const float *__restrict x, size_t n);
void vexpm1f(float *__restrict y, const float *__restrict x, size_t n){size_t start=0; for(size_t i=0;i<n;i++) if(isfinite(x[i]) && (fabsf(x[i]) < 0x1p-120f)) {
 if(i>start) a210_scalar_vexpm1f(y+start, x+start,i-start);
 a210_rvv_vexpm1f(y+i, x+i, 1); start=i+1;
 } if(start<n) a210_scalar_vexpm1f(y+start, x+start,n-start);}
extern void a210_scalar_vlogf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vlogf(float *__restrict y, const float *__restrict x, size_t n);
void vlogf(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vlogf(y, x, n);}
extern void a210_scalar_vlog2f(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vlog2f(float *__restrict y, const float *__restrict x, size_t n);
void vlog2f(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vlog2f(y, x, n);}
extern void a210_scalar_vlog10f(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vlog10f(float *__restrict y, const float *__restrict x, size_t n);
void vlog10f(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vlog10f(y, x, n);}
extern void a210_scalar_vlog1pf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vlog1pf(float *__restrict y, const float *__restrict x, size_t n);
void vlog1pf(float *__restrict y, const float *__restrict x, size_t n){size_t start=0; for(size_t i=0;i<n;i++) if(isfinite(x[i]) && (fabsf(x[i]) < 0x1p-120f)) {
 if(i>start) a210_scalar_vlog1pf(y+start, x+start,i-start);
 a210_rvv_vlog1pf(y+i, x+i, 1); start=i+1;
 } if(start<n) a210_scalar_vlog1pf(y+start, x+start,n-start);}
extern void a210_scalar_vlog2p1f(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vlog2p1f(float *__restrict y, const float *__restrict x, size_t n);
void vlog2p1f(float *__restrict y, const float *__restrict x, size_t n){size_t start=0; for(size_t i=0;i<n;i++) if(isfinite(x[i]) && (fabsf(x[i]) < 0x1p-60f)) {
 if(i>start) a210_scalar_vlog2p1f(y+start, x+start,i-start);
 a210_rvv_vlog2p1f(y+i, x+i, 1); start=i+1;
 } if(start<n) a210_scalar_vlog2p1f(y+start, x+start,n-start);}
extern void a210_scalar_vpowf(float *__restrict y, const float *__restrict ax,
	    const float *__restrict bx, size_t n);
extern void a210_rvv_vpowf(float *__restrict y, const float *__restrict ax,
	    const float *__restrict bx, size_t n);
void vpowf(float *__restrict y, const float *__restrict ax,
	    const float *__restrict bx, size_t n){size_t start=0; for(size_t i=0;i<n;i++) if(isfinite(ax[i]) && (fabsf(ax[i]) < 0x1p-120f || fabsf(ax[i]) > 0x1p120f)) {
 if(i>start) a210_scalar_vpowf(y+start, ax+start, bx+start,i-start);
 a210_rvv_vpowf(y+i, ax+i, bx+i, 1); start=i+1;
 } if(start<n) a210_scalar_vpowf(y+start, ax+start, bx+start,n-start);}
extern void a210_scalar_vsqrtf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vsqrtf(float *__restrict y, const float *__restrict x, size_t n);
void vsqrtf(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vsqrtf(y, x, n);}
extern void a210_scalar_vrsqrtf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vrsqrtf(float *__restrict y, const float *__restrict x, size_t n);
void vrsqrtf(float *__restrict y, const float *__restrict x, size_t n){a210_rvv_vrsqrtf(y, x, n);}
extern void a210_scalar_vcbrtf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vcbrtf(float *__restrict y, const float *__restrict x, size_t n);
void vcbrtf(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vcbrtf(y, x, n);}
extern void a210_scalar_vhypotf(float *__restrict y, const float *__restrict ax,
	      const float *__restrict bx, size_t n);
extern void a210_rvv_vhypotf(float *__restrict y, const float *__restrict ax,
	      const float *__restrict bx, size_t n);
void vhypotf(float *__restrict y, const float *__restrict ax,
	      const float *__restrict bx, size_t n){a210_rvv_vhypotf(y, ax, bx, n);}
extern void a210_scalar_verff(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_verff(float *__restrict y, const float *__restrict x, size_t n);
void verff(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_verff(y, x, n);}
extern void a210_scalar_verfcf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_verfcf(float *__restrict y, const float *__restrict x, size_t n);
void verfcf(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_verfcf(y, x, n);}
extern void a210_scalar_vtgammaf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vtgammaf(float *__restrict y, const float *__restrict x, size_t n);
extern float newlib_scalar_tgammaf(float);
void vtgammaf(float *__restrict y, const float *__restrict x, size_t n){for(size_t i=0;i<n;i++) { float a=x[i];
 if(!isfinite(a) || a>32 || a < -32 || (a<=0 && a==(int)a) || fabs(a)<0x1p-100) a210_rvv_vtgammaf(y+i,x+i,1);
 else y[i]=newlib_scalar_tgammaf(a); }}
extern void a210_scalar_vlgammaf(float *__restrict y, const float *__restrict x, size_t n);
extern void a210_rvv_vlgammaf(float *__restrict y, const float *__restrict x, size_t n);
void vlgammaf(float *__restrict y, const float *__restrict x, size_t n){a210_scalar_vlgammaf(y, x, n);}
extern void a210_scalar_vlgamma_rf(float *__restrict y, int *__restrict sign,
		 const float *__restrict x, size_t n);
extern void a210_rvv_vlgamma_rf(float *__restrict y, int *__restrict sign,
		 const float *__restrict x, size_t n);
void vlgamma_rf(float *__restrict y, int *__restrict sign,
		 const float *__restrict x, size_t n){a210_rvv_vlgamma_rf(y, sign, x, n);}