#include"nn.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

//y=Ax+bの計算をする
//m,nは行列とベクトルの配列
void fc(int m, int n, const float * x, const float * A, const float * b, float * y){
    for(int i=0;i<m;i++){
        float Ax=0;
        for(int j=0;j<n;j++){
            Ax = Ax + A[i*n+j]*x[j];
        }
        y[i] = Ax + b[i];
    }
}
//n行の列ベクトルxに対し式(2)を計算しyに書き込む
void relu(int n,const float *x,float *y){
    for(int i=0;i<n;i++){
        if(x[i]>0){
            y[i] = x[i];
        }else{
            y[i] = 0;
        }
    }
}
//n行の列ベクトルxに対し式(4)を計算しyに書き込む
void softmax(int n,const float*x,float *y){
    int i;
    float x_max =0;
    for(i=0;i<n;i++){
        if(y[i]>x_max){
            x_max = y[i];
        }
    }
    float sumexp=0;
    for(i=0;i<n;i++){
        sumexp = sumexp + exp(x[i]-x_max);
    }
    for(i=0;i<n;i++){
        y[i] = exp(x[i]-x_max)/sumexp;
    }
}

//あらかじめ用意されたパラメータであるA1,b1,A2,b2,A3,b3,xに対しNNによる推論を行い
//yの各要素のうちの最大値を取る添え字ansを戻り値として返す
int inference6 (const float *A1,const float *b1,const float *A2,const float *b2, const float *A3,const float *b3,const float *x,float *y){
    float *y1 =malloc(sizeof(float)*50);
    float *y2 =malloc(sizeof(float)*100);
    fc(50,784,x,A1,b1,y1);
    relu(50,y1,y1);
    fc(100,50,y1,A2,b2,y2);
    relu(100,y2,y2);
    fc(10,100,y2,A3,b3,y);
    softmax(10,y,y);
    float max=0;
    int ans=0;
    for(int i=0;i<10;i++){
        if(max<y[i]){
            max = y[i];
            ans = i;
        }
    }
    return ans;
}

void load(const char *filename, int m, int n, float *A, float *b)
{
    FILE *fp;
    fp = fopen(filename, "rb");
    fread(A, sizeof(float), n * m, fp);
    fread(b, sizeof(float), m , fp);
    fclose(fp);
}

int main(int argc, char * argv[]) {
    float*train_x=NULL;
    unsigned char * train_y=NULL;
    int train_count=-1;
    float *test_x=NULL;
    unsigned char *test_y=NULL;
    int test_count=-1;
    int width=-1;
    int height=-1;
    load_mnist(&train_x,&train_y,&train_count,&test_x,&test_y,&test_count,&width,&height);
    float * A1 = malloc(sizeof(float) *784*50);
    float * b1 = malloc(sizeof(float) *50);
    float * A2 = malloc(sizeof(float) *50*100);
    float * b2 = malloc(sizeof(float) *100);
    float * A3 = malloc(sizeof(float) *100*10);
    float * b3 = malloc(sizeof(float) *10);
    float * x = load_mnist_bmp(argv[4]);
    float *y = malloc(sizeof(float) * 10);
    load(argv[1], 50, 784, A1, b1);
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);
    int a;
    scanf("%d", &a);
    printf("The answer is %d correct answer is %d\n", inference6(A1, b1, A2, b2, A3, b3, x, y), test_y[a]);
    return 0;
}