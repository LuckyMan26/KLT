// KLT.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <map>
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp> 
std::map <std::pair<int,int>, double> map_of_eigen_value;

using Eigen::MatrixXd;
MatrixXd get_window(int window_size, cv::Mat& img, int x, int y) {
    
    int startX = x - window_size / 2;
    int startY = y - window_size / 2;
    MatrixXd res(window_size, window_size);
   
    
  
    
    for (int i = startX; i < startX + window_size; i++) {
        for (int j = startY; j < startY + window_size; j++) {
                if (i < 0) {
                    if (j < 0) {
                        res(std::abs(i-startX), std::abs(j - startY)) = img.at<uchar>(0, 0);
                    }
                    else {
                       
                        res(std::abs(i - startX),j - startY) = (img.at<uchar>(0, std::min(j,img.cols-1)));
                    }
                }
                else if (j < 0) {
                    if (i < 0) {
                        res(std::abs(i - startX), std::abs(j - startY)) = img.at<uchar>(0, 0);
                    }
                    else {
                        res(i-startX, std::abs(j - startY)) = img.at<uchar>(std::min(img.rows-1,i), 0);
                    }
                }
                else {
                    
                    res(i-startX, j-startY) = img.at<uchar>(std::min(i,img.rows-1),  std::min(j,img.cols-1));
                }
            
            
        }
    }
    return res;
}
MatrixXd image_derivitive_x(cv::Mat& img, cv::Point2f p) {
    int window_size = 3;
     MatrixXd m1 = get_window(3,img,p.x+1,p.y);
     MatrixXd m2 = get_window(3, img, p.x - 1, p.y);
     MatrixXd res(3, 3);

     res = (m1 - m2)/2;

     return res;
}

MatrixXd image_derivitive_y(cv::Mat& img, cv::Point2f p) {
    int window_size = 3;
    MatrixXd m1 = get_window(3, img, p.x, p.y+1);
    MatrixXd m2 = get_window(3, img, p.x, p.y-1);
    MatrixXd res(3, 3);

    res = (m1 - m2)/2;

    return res;
}
double compute_I_square(MatrixXd I) {
    double res = 0;
    for (int i = 0; i < I.rows(); i++) {
        for (int j = 0; j < I.cols(); j++) {
            res += I(i, j) * I(i, j);
        }
    }
    return res;
}
double compute_I_xy(MatrixXd I_x, MatrixXd I_y) {
    double res = 0;
    if (I_x.rows() != I_y.cols() && I_x.rows() != I_x.cols()) {
        return 0;
    }
    else {
        for (int i = 0; i < I_x.rows(); i++) {
            for (int j = 0; j < I_y.cols(); j++) {
                res += I_x(i, j) * I_y(i, j);
            }
        }
    }
    return res;
}
MatrixXd compute_h_matrix(MatrixXd I_x, MatrixXd I_y,int window_size) {
    MatrixXd h(2, 2);
    h(0, 0) = compute_I_square(I_x);
    h(1, 1) = compute_I_square(I_y);
    h(0, 1) = compute_I_xy(I_x, I_y);
    h(1, 0) = h(0, 1);
    return h;
}

double get_eigen_values(MatrixXd h) {
    int threshold = 100;
    Eigen::EigenSolver<Eigen::MatrixXd> solver(h);

    Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
    
    Eigen::VectorXd eigenvaluesimg = solver.eigenvalues().imag();

    if (eigenvaluesimg.maxCoeff() != 0 && eigenvaluesimg.minCoeff() != 0) {
        return  std::numeric_limits<double>::min();
    }

    double maxEigenvalue = eigenvalues.maxCoeff();
    double minEigenvalue = eigenvalues.minCoeff();
   
    return (minEigenvalue);
    
   
}
void get_eigen_for_pixel(cv::Mat& img,cv::Point2f p) {
    MatrixXd Ix= image_derivitive_x(img,  p);
   
    MatrixXd Iy = image_derivitive_y(img, p);
   
    MatrixXd h = compute_h_matrix(Ix, Iy, 3);
    double eigen = get_eigen_values(h);
    map_of_eigen_value[std::make_pair(p.x,p.y)] = eigen;
    
    
}
bool check_is_corner(cv::Mat& img, cv::Point2f p,double max) {
    double eigen = map_of_eigen_value[std::make_pair(p.x, p.y)];
    if (eigen > 0.05 * max) {
       
            return true;
         

        
    }
    return false;
}
cv::Mat get_next_image_pyramidial(cv::Mat& I) {
    int nx = (I.cols + 1) / 2;
    int ny = (I.rows + 1) / 2;
   // std::cout << nx << " " << ny << std::endl;
    cv::Mat res =  cv::Mat::zeros(ny, nx, CV_8UC1);
    
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
           
            res.at<uchar>(i,j) = 0.25 * I.at<uchar>(2 * i, 2 * j) + 0.125 * (I.at<uchar>(std::max(2 * i - 1,0), 2 * j) + 
                I.at<uchar>(std::min(2 * i + 1,ny-1), 2 * j) + I.at<uchar>(2 * i, std::max(2 * j - 1,0)) +
                I.at<uchar>(2 * i, std::min(2 * j + 1,nx-1))) +
                0, 0625 * (I.at<uchar>(std::max(2 * i - 1,0), std::max(2 * j - 1,0)) +
                    I.at<uchar>(std::min(2 * i + 1,ny-1), std::min(2 * j + 1,nx)) +
                    I.at<uchar>(std::max(2 * i - 1,0), std::min(2 * j + 1,nx-1)) +
                    I.at<uchar>(std::min(2 * i + 1,ny-1), std::max(2 * j - 1,0)));
         
        }
    }
    return res;
}
std::vector<cv::Mat> build_pyramidial_representation(cv::Mat& I) {

    std::vector<cv::Mat> res;
    res.push_back(I);
    for (int i = 1; i <= 4; i++) {
        res.push_back(get_next_image_pyramidial(res[i - 1]));
       
    }

    return res;
}
MatrixXd image_difference(cv::Mat& I, cv::Mat& J, int x, int y, int g_x, int g_y, int mu_x, int mu_y) {
    MatrixXd matrix_I = get_window(3, I, x, y);
    MatrixXd matrix_J = get_window(3, J, x + g_x + mu_x, y + g_y + mu_y);
   // std::cout << "Matrix_I " << matrix_I.rows() << " " << matrix_I.cols() << "\n";
   // std::cout << "Matrix_J " << matrix_J.rows() << " " << matrix_J.cols() << "\n";
   // std::cout << "Result (Matrix_I - Matrix_J):\n" << matrix_I - matrix_J << "\n";
   
    
    return Eigen::MatrixXd(matrix_I - matrix_J);

}
Eigen::VectorXd mismatch_vector(MatrixXd image_difference, MatrixXd I_x, MatrixXd I_y, int x, int y) {
    Eigen::VectorXd b_k(2);
    b_k(0) = 0;
    b_k(1) = 0;
    for (int i = 0; i < I_x.rows(); i++) {
        for (int j = 0; j < I_x.cols(); j++) {
            //std::cout << I_x(i, j) << " ";
            b_k(0) += I_x(i, j) * image_difference(i, j);
            b_k(1) += I_y(i, j) * image_difference(i, j);

        }
    }
    return Eigen::VectorXd(b_k);
}
cv::Point2f find_corresponding_point(cv::Mat& I, cv::Mat& J, cv::Point2f u) {
    std::vector<cv::Mat> I_l =  build_pyramidial_representation(I);
    std::vector<cv::Mat> J_l = build_pyramidial_representation(J);
    Eigen::VectorXd g(2);
    g(0) = 0;
    g(1) = 0;
    Eigen::VectorXd d(2);
    d(0) = 0;
    d(1) = 0;
    Eigen::VectorXd v(2);
    for (int i = I_l.size()-1; i >= 0; i--) {
        int x_L = u.x / std::pow(2, i);
        int y_L = u.y / std::pow(2, i);
        MatrixXd I_x_L = image_derivitive_x(I_l[i],cv::Point2f(x_L,y_L));
        MatrixXd I_y_L = image_derivitive_y(J_l[i], cv::Point2f(x_L, y_L));

        MatrixXd G = compute_h_matrix(I_x_L, I_y_L, 3);
       
        //std::cout << "Hessian: " << G << std::endl;
        Eigen::FullPivLU<Eigen::MatrixXd> lu_decomposition(G);

      
        if (!lu_decomposition.isInvertible()) {
            
           return cv::Point2f(std::numeric_limits<double>::min(), std::numeric_limits<double>::min());
        }
        MatrixXd G_inverse = G.inverse();
       
       
        v(0) = 0;
        v(1) = 0;
        Eigen::VectorXd eta_k;

        for (int k = 1; k <= 5 && eta_k.norm() < 0.03; k++) {
            MatrixXd img_diff = image_difference(I_l[i], J_l[i], x_L, y_L, g(0), g(1), v(0), v(1));
           
            Eigen::VectorXd b_k = mismatch_vector(img_diff, I_x_L, I_y_L, x_L, y_L);
           
            eta_k  = G_inverse * b_k;
          
            v = eta_k + v;
        }
        d = v;
        g = 2 * (g + d);
    }

    d = g + d;
    
    cv::Point2f point_v = cv::Point2f(u.x + d(0), u.y + d(1));
    if (point_v.x < 0) {
        point_v.x = 0;
    }
    if (point_v.y < 0) {
        point_v.y = 0;
    }
    if (point_v.x > J.rows -1) {
        point_v.x = J.rows - 1;
    }
    if (point_v.y > J.cols - 1) {
        point_v.y = J.cols - 1;
    }
    return point_v;
}
int main()
{

   
   
 
    double maxValue = std::numeric_limits<double>::min(); 

    
    std::cout <<"Max: " << maxValue << "\n";
  
  
    
    int count = 0;
    cv::VideoCapture cap("E:\\test3.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file." << std::endl;
        return -1;
    }
   
    
        cv::Mat cur_frame,next_frame;
        cv::Mat cur_frame_grayscale, next_frame_grayscale;
        cap >> cur_frame >> next_frame; 
        count++;
   
        cv::cvtColor(cur_frame, cur_frame_grayscale, cv::COLOR_BGR2GRAY);
        cv::cvtColor(next_frame, next_frame_grayscale, cv::COLOR_BGR2GRAY);
        
       
        for (int i = 0; i < cur_frame_grayscale.rows; i++) {
            for (int j = 0; j < cur_frame_grayscale.cols; j++) {
                get_eigen_for_pixel(cur_frame_grayscale, cv::Point2f(i, j));
            }
        }
        for (auto i = map_of_eigen_value.begin(); i != map_of_eigen_value.end(); i++) {
            //std::cout << i->second << "\n";
            if (maxValue < i->second) {
                maxValue = i->second;
            }

        }
        std::vector<cv::Point2f> corners;
        for (int i = 0; i < cur_frame_grayscale.rows; i++) {
            for (int j = 0; j < cur_frame_grayscale.cols; j++) {

                if (check_is_corner(cur_frame_grayscale, cv::Point2f(i, j), maxValue)) {
                    corners.push_back(cv::Point2f(i, j));


                }
            }
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frameSize(
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
       
        cv::VideoWriter outputVideo("result.mp4", cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, frameSize);
        std::vector<std::vector<cv::Point2f>> displacement;
        for (int i = 0; i < corners.size(); i++) {
            cv::Point2f v = find_corresponding_point(cur_frame_grayscale, next_frame_grayscale, corners[i]);
            if (v == cv::Point2f(std::numeric_limits<double>::min(), std::numeric_limits<double>::min())) {
                corners.erase(corners.begin() + i);
            }
            else {
                cv::circle(cur_frame, corners[i], 1, cv::Scalar(0, 255, 0), -1);
                //std::cout << "Corresponding point: " << v << std::endl;
                corners[i] = v;
                
            }


        }
        cv::Mat img_to_write;
        outputVideo.write(cur_frame);

       
        
       
        std::clock_t clock = std::clock();
        while (true) {
            count++;
            std::cout << count << "\n";
            cur_frame = next_frame;
            cap >> next_frame;
            std::vector<cv::Point2f> vec_of_dicplacement;
            for (int i = 0; i < corners.size(); i++) {
                
                cv::Point2f v = find_corresponding_point(cur_frame_grayscale, next_frame_grayscale, corners[i]);
                if (v == cv::Point2f(std::numeric_limits<double>::min(), std::numeric_limits<double>::min())) {
                   corners.erase(corners.begin()+i);
               }
                //std::cout << "Corresponding point: " << v << std::endl;
                else {
                    corners[i] = v;
                    vec_of_dicplacement.push_back(v);
                }


            }
            displacement.push_back(vec_of_dicplacement);

           
            if (cur_frame.empty() || next_frame.empty()) {
                break;
            }
           
           

           
        }
       
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        
        cv::Mat output;
        int temp = 0;
        while (true) {
            cv::Mat frame;
           
            cap >> frame;
            if (frame.empty()) {
                break;
            }
            output = frame;

            for (int j = 0; j < displacement.size(); j++) {
                for (int k = 0; k < std::min(std::size_t(temp), displacement[j].size()); k++) {
                    cv::circle(output, displacement[j][k], 1, cv::Scalar(0, 0, 255), -1);
                }
               
            }
            outputVideo.write(output);
            temp++;
        }
        
        outputVideo.release();
        cap.release();

       
      
        
        cv::destroyAllWindows();
}

