#include"nn.h"
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
//m*n行列xを表示
void print(int m, int n, const float*x){
    int i,j;
    for(i=0;i<=m-1;i++){
        for(j=0;j<=n-1;j++){
            printf("%f ",x[j+n*i]);
        }
        printf("\n");
    }
}
//n行の列ベクトルxをyにコピー
void copy(int n, const float *x, float *y){
    int i;
    for (i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}
//y=Ax+bの計算
//m,nは行列とベクトルの配列を表す
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
//n行の列ベクトルyに対し式(11)を計算しdxに書き込む
//tは正解を表す[0:9]の整数
void softmaxwithloss_bwd(int n, const float *y,unsigned char t,float *dx){
    int i;
    for(i=0;i<=n-1;i++){
        if(i==t){
            dx[i]=y[i]-1;
        } else {
            dx[i]=y[i];
        }
    }
}
//n行の列ベクトルdyに対し式(13)を計算しdxに書き込む
//xは順伝播のひとつ前の層で得た計算結果と同値
void relu_bwd(int n,const float*x,const float *dy,float *dx){
    int i;
    for(i=0;i<=n-1;i++){
        if(x[i]>0){
            dx[i]=dy[i];
        } else {
            dx[i]=0;
        }
    }
}
//式(11)を計算しdA,db,dxに書き込む
//dx,dA,dbはそれぞれx,A,bと同じ大きさの配列
//m,nは行列とベクトルの配列の大きさ
void fc_bwd(int m,int n,const float *x,const float *dy,const float *A,float *dA,float *db,float *dx){
    int i,j;
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            dA[n*i+j]=dy[i]*x[j];
        }
    }
    copy(m,dy,db);
    for(i=0;i<n;i++){
        dx[i]=0;
        for(j=0;j<m;j++){
            dx[i]+=A[n*j+i]*dy[j];
        }
    }
}

//n個の要素を持つxを乱数を利用して入れ替える
void shuffle(int n,int *x){
    int i,j,b;
    b=0;
    for(i=0;i<n;i++){
        j=rand()%n;
        b=x[j];
        x[j]=x[i];
        x[i]=b;
    }
}
//y,tを入力して式(6)を計算し損失関数を表す変数に書き込む
//ここでtは1である
float loss(const float y,unsigned char t){
    return(float)  (-1 * t * log(y + 0.0000001));
}
//n個の要素を持つ配列oにxを加える
void add(int n,const float*x,float*o){
    int i;
    for(i=0;i<=n-1;i++){
        o[i]+=x[i];
    }
}
//n個の要素を持つ配列oをx倍する
void scale(int n,float x,float*o){
    int i;
    for(i=0;i<=n-1;i++){
        o[i]*=x;
    }
}
//n個の要素を持つ配列oにxを代入する
void init(int n,float x,float *o){
    int i;
    for(i=0;i<=n-1;i++){
        o[i]=x;
    }
}
//n個の要素を持つ配列oを[-1:1]の範囲内の乱数で初期化する
void rand_init(int n,float*o){
    int i;
    srand(time(NULL));
    for(i=0;i<=n-1;i++){
        o[i]=1-2*(double)rand()/(double)(RAND_MAX);
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
//あらかじめ用意されたパラメータであるA1,b1,A2,b2,A3,b3,x,tに対し誤差逆伝播を行いdA1,db1,dA2,db2,dA3,db3に書き込む
//tは正解を表す[0:9]の整数
void backward6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3,float *x, unsigned char t,float *dA1,  float * db1,float *dA2,  float * db2,float *dA3,  float * db3){
    float fc1[784];//最下流の入力
    float ReLU1[50];
    float fc2[50];
    float ReLU2[100];
    float fc3[100];
    float dx_fc1[784];
    float dx_relu1[50];
    float dx_fc2[50];
    float dx_relu2[100];
    float dx_fc3[100];
    float dx_soft[10];
    float y[10];
    copy(784, x, fc1);
    fc(50, 784, fc1, A1, b1, ReLU1);
    relu(50, ReLU1, fc2);
    fc(100, 50, fc2, A2, b2, ReLU2);
    relu(100, ReLU2, fc3);
    fc(10, 100, fc3, A3, b3, y);
    softmax(10, y, y);
     
    softmaxwithloss_bwd(10, y, t, dx_soft);
    fc_bwd(10, 100, fc3, dx_soft, A3, dA3, db3, dx_fc3);
    relu_bwd(100, ReLU2, dx_fc3, dx_relu2);
    fc_bwd(100, 50, fc2, dx_relu2, A2, dA2, db2, dx_fc2);
    relu_bwd(50, ReLU1, dx_fc2, dx_relu1);
    fc_bwd(50, 784, fc1, dx_relu1, A1, dA1, db1, dx_fc1);
}
//filenameをファイル名とし学習したパラメータA,bをファイルに書き込む
void save(const char *filename, int n, int m, const float *A,const float *b){
    FILE *file;
    file = fopen(filename, "wb");
    if(file == NULL){
        printf("error");
    }else{
        fwrite(A, sizeof(float) ,n*m, file);
        fwrite(b, sizeof(float), n, file);
        fclose(file);
    }
}

int main(int argc,char *argv[]){
    srand(time(NULL));
    float*train_x=NULL;
    unsigned char * train_y=NULL;
    int train_count=-1;
    float *test_x=NULL;
    unsigned char *test_y=NULL;
    int test_count=-1;
    int width=-1;
    int height=-1;
    load_mnist(&train_x,&train_y,&train_count,&test_x,&test_y,&test_count,&width,&height);
    float *y1 = malloc(sizeof(float) * 50);
    float *dA1 = malloc(sizeof(float) * 784 * 50);
    float *db1 = malloc(sizeof(float) * 50);
    float *dA2 = malloc(sizeof(float) * 50 * 100);
    float *db2 = malloc(sizeof(float) * 100);
    float *dA3 = malloc(sizeof(float) * 100 * 10);
    float *db3 = malloc(sizeof(float) * 10);
    float *avg_dA1= malloc(sizeof(float) * 784 * 50);
    float *avg_db1 = malloc(sizeof(float) * 50);
    float *avg_dA2 = malloc(sizeof(float) * 50 * 100);
    float *avg_db2 = malloc(sizeof(float) * 100);
    float *avg_dA3 = malloc(sizeof(float) * 100 * 10);
    float *avg_db3 = malloc(sizeof(float) * 10); 
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *b2 = malloc(sizeof(float) * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b3 = malloc(sizeof(float) * 10);
    rand_init(784 * 50, A1);
    rand_init(50, b1);
    rand_init(50 * 100 , A2);
    rand_init(100, b2);
    rand_init(100 * 10, A3);
    rand_init(10, b3);
    int mini=100;//ミニバッチサイズ
    int N=train_count;
    float  study= 0.01;//学習率
    int index[train_count];
    int sum=0;//正解数
    float loss1;//損失関数
    int epo=10;//エポック回数
    int i,j,k,l;

    for (i = 0; i < train_count;i++) {
        index[i] = i;
    }

    for (l=0;l<epo;l++) {
        shuffle(train_count, index);
        //式(8)の計算(勾配法によってパラメータを決定）  
        for (k = 0; k < N/mini; k++) {
            //0で初期化
            init(784*50, 0.0, avg_dA1);
            init(50, 0.0, avg_db1);
            init(50*100, 0.0, avg_dA2);
            init(100, 0.0, avg_db2);
            init(100*10, 0.0, avg_dA3);
            init(10, 0.0, avg_db3);
            
            for (j=0;j<mini; j++) {
                //勾配を計算し平均勾配に加算
                backward6(A1, b1, A2, b2, A3, b3, train_x + width*height*index[j+k*mini], train_y[index[j + k*mini]],dA1, db1, dA2,db2,dA3,db3);    
                add(784*50, dA1, avg_dA1);
                add(50, db1, avg_db1);
                add(50*100, dA2, avg_dA2);
                add(100, db2, avg_db2);
                add(100*10, dA3, avg_dA3);
                add(10, db3, avg_db3);   
            }
            //平均勾配をミニバッチサイズで割る
            scale(784*50, 0.01, avg_dA1);
            scale(50, 0.01, avg_db1);
            scale(50*100, 0.01, avg_dA2);
            scale(100, 0.01, avg_db2);
            scale(100*10, 0.01, avg_dA3);
            scale(10, 0.01, avg_db3);
            //平均勾配に-1*(学習率)をかける
            scale(784*50, -study, avg_dA1);
            scale(50, -study, avg_db1);
            scale(50*100, -study, avg_dA2);
            scale(100, -study, avg_db2);
            scale(100*10, -study, avg_dA3);
            scale(10, -study, avg_db3);
            //A1,A2,A3,b1,b2,b3に加算
            add(784*50, avg_dA1, A1);
            add(50, avg_db1, b1);
            add(50*100, avg_dA2, A2);
            add(100, avg_db2, b2);
            add(100*10, avg_dA3, A3);
            add(10, avg_db3, b3);
        }
        //損失関数、正解率をリセット
        sum=0;
        loss1=0;
        //損失関数、正解率を計算し表示
        for (i = 0; i<test_count;i++) {
            if (inference6(A1,b1,A2,b2,A3,b3,test_x+width*height*i,y1) == test_y[i]) {
            sum++;
            }
        loss1+= loss(y1[test_y[i]], 1);
        }
        printf("test %d epoc %f ", l+1, loss1/test_count);
        printf("%f%%\n", 100.0*sum/test_count );
        //損失関数、正解率をリセット
        sum=0;
        loss1=0;
        //損失関数、正解率を計算し表示
        for (i = 0; i<train_count;i++) {
            if (inference6(A1,b1,A2,b2,A3,b3,train_x+width*height*i,y1) == train_y[i]) {
            sum++;
            }
        loss1+= loss(y1[train_y[i]], 1);
        }
        printf("train %d epoc %f ", l+1, loss1/train_count);
        printf("%f%%\n", 100.0*sum/train_count );
    }
    //学習したパラメータを実行時引数で入力したファイルに保存
    save(argv[1],50,784, A1, b1);
    save(argv[2],100,50, A2, b2);
    save(argv[3],10,100, A3, b3);
    return 0;
}