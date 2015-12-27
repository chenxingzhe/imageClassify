#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <string>
#include <stdio.h>
#include <iostream>
using namespace cv;
using namespace std;
int main()
{
    double time = (double)cvGetTickCount();
    string classnum[5];
    classnum[0]="./elephant/image_";
    classnum[1]="./butterfly/image_";
    classnum[2]="./cup/image_";
    classnum[3]="./camera/image_";
    classnum[4]="./crocodile_head/image_";
    vector<Mat> imageall;
    Mat data;
    const int trainnum=45;
    const int desnum=30;
const int k=200;
    for(int i=0;i<5;i++)
    for(int j=1;j<=trainnum;j++)
    {
        string num;
        char numm[10];
        sprintf(numm,"%04d.jpg",j);
        num=numm;
        string name=classnum[i]+num;
        cout<<name<<endl;
        Mat image1=imread(name,0);
       imshow("1",image1);
        vector<KeyPoint> keypoints1,keypoints2;   
        //在这里切换不同的detector
        //OrbFeatureDetector detector(200);
        SurfFeatureDetector detector(desnum);
        //FastFeatureDetector detector(25);
        //SurfFeatureDetector detector(50);
        //detector.detect(reimage1, keypoints1);
        //detector.detect(reimage2, keypoints2);
        detector.detect(image1, keypoints1);
        // 描述特征点
        //这里选择不同的descriptor
        //OrbDescriptorExtractor Desc;
        //SurfDescriptorExtractor Desc;
        SiftDescriptorExtractor Desc;
        //BriefDescriptorExtractor Desc;
        Mat descriptros1,descriptros2;
        //Desc.compute(reimage1,keypoints1,descriptros1);
        //Desc.compute(reimage2,keypoints2,descriptros2);
        Desc.compute(image1,keypoints1,descriptros1);
        descriptros1=descriptros1*100;

        //cout<<descriptros2<<endl;
        int height1=descriptros1.rows;
        int width1=descriptros1.cols;
        // 计算匹配点数  
        data.push_back(descriptros1);
        imageall.push_back(descriptros1);
    }
cout<<data.size()<<endl;
const int attempts=3;
Mat label,centers;
kmeans(data,k,label,TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0), attempts, KMEANS_PP_CENTERS, centers);
cout<<imageall.size()<<endl;
cout<<centers.size()<<endl;
cout<<label.size()<<endl;
int count=0;
Mat feature;
Mat fealabel;
for(int i=0;i<imageall.size();i++)
{
    float fea[k];
    memset(fea,0,sizeof(fea));
    //cout<<imageall[i].cols<<endl;
    for(int j=0;j<imageall[i].rows;j++)
    {
        fea[label.at<int>(count++)]++;
    }
    for(int j=0;j<k;j++)
    {
        fea[j]/=imageall[i].rows;
    }
    //cout<<fea[0]<<endl;
    feature.push_back(Mat(1,k,CV_32FC1,fea));
    fealabel.push_back(Mat(1,1,CV_32FC1,i/trainnum));
}
cout<<feature<<endl;
cout<<centers<<endl;
CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::RBF;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 2000, 1e-6);

    // 对SVM进行训练
    CvSVM SVM;
    SVM.train(feature, fealabel, Mat(), Mat(), params);
   for(int i=0;i<5*trainnum;i++)
   {
      // cout<<feature.row(i)<<endl;
   cout<<i/trainnum<<" "<<SVM.predict(feature.row(i))<<endl;
   }
CvKNearest knn;  
    float testnum[k];
    for(int i=0;i<k;i++)
    {
        testnum[i]=i;
    }
    Mat labelsMat(1,k,CV_32FC1,testnum);
    //cout<<centers<<endl;
    knn.train(centers,labelsMat,Mat(), false, 1 );  
    int countnum=0;
    for(int i=0;i<5;i++)
    for(int j=46;j<=50;j++)
    {
        string num;
        char numm[10];
        sprintf(numm,"%04d.jpg",j);
        num=numm;
        string name=classnum[i]+num;
        cout<<name<<endl;
        Mat image1=imread(name,0);
       imshow("1",image1);
        vector<KeyPoint> keypoints1,keypoints2;   
        //在这里切换不同的detector
        //OrbFeatureDetector detector(200);
        SurfFeatureDetector detector(desnum);
        //FastFeatureDetector detector(25);
        //SurfFeatureDetector detector(50);
        //detector.detect(reimage1, keypoints1);
        //detector.detect(reimage2, keypoints2);
        detector.detect(image1, keypoints1);
        // 描述特征点
        //这里选择不同的descriptor
        //OrbDescriptorExtractor Desc;
        //SurfDescriptorExtractor Desc;
        SiftDescriptorExtractor Desc;
        //BriefDescriptorExtractor Desc;
        Mat descriptros1,descriptros2;
        //Desc.compute(reimage1,keypoints1,descriptros1);
        //Desc.compute(reimage2,keypoints2,descriptros2);
    Desc.compute(image1,keypoints1,descriptros1);
        descriptros1=descriptros1*100;
        float fea[k];
        memset(fea,0,sizeof(fea));
        for(int ii=0;ii<descriptros1.rows;ii++)
        {
            Mat testmat=descriptros1.row(ii).clone();
            testmat.convertTo(testmat,CV_32FC1);
            float ttt[128];
            for(int jj=0;jj<128;jj++)
            {
                ttt[jj]=testmat.at<float>(jj);
            }
           // cout<<testmat<<endl;
            float nn=knn.find_nearest(Mat(1,128,CV_32FC1,ttt),1);
            fea[(int)nn]++;
        }
        Mat test(1,k,CV_32FC1,fea);
       
       test=test/descriptros1.rows;
    // cout<<test<<endl;
        int testpre=SVM.predict(test);
        if(testpre==i)
        {
            countnum++;
            cout<<"yes"<<endl;
        }
    else
    {
        cout<<"no"<<endl;
    }
    }
    cout<<"accuracy:"<<countnum*1.0/25<<endl;
waitKey();


    return 0;

}
