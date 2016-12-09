#include <iostream>
#include <cmath>
#include <armadillo>
#include <string>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sstream>


using namespace arma;
using namespace std;

const double PI  =3.141592653589793238463;

double uExact1D(double x, double t) {
    // The exact solution for up to N=3 in the fourier series
    return x+(2/PI)*(-sin(PI*x)*exp(-PI*PI*t) + 0.5*sin(2*PI*x)*exp(-4*PI*PI*t) - (1/3)*sin(3*PI*x)*exp(-9*PI*PI*t));
}

double uExact2D(double x, double y, double t) {
    return sin(PI*x)*sin(PI*y)*exp(-2*PI*PI*t);
}

double initialCondition(double x, double y) {
    return sin(PI*x)*cos(PI*y-PI/2);
}

vec forwardExplicit1D(double dx=0.2, double dt=0.01, double T=1.0, bool writeToFile=false) {
    if(dt > dx*dx/2) {
        cout << "Stability requirement not met. Please set dt < dx*dx." << endl;
    }

    int Nx = 1.0/dx + 1;
    int Nt = T/dt + 1;

    double constant = (dt)/(dx*dx);

    ofstream file;
    if(writeToFile) {
        string filename = "ForwardExplicit1D.txt";

        file.open(filename);
    }

    vec u = zeros(Nx); //Current timestep
    vec u1 = zeros(Nx); //Previous timestep

    if(writeToFile) {
        for(int k = 0; k < Nx; k++) {
            file <<  u1(k) << " ";
        }
        file << "\n";
    }

    for(int n = 1; n < Nt - 1; n++) {
        for(int i = 1; i < Nx - 1; i++) {
            u(i) = constant*(u1(i+1)-2*u1(i)+u1(i-1)) + u1(i);
        }
        // Insert boundary conditions
        u(0) = 0.0; u(Nx-1) = 1.0;

        if(writeToFile) {
            for(int k = 0; k < Nx; k++) {
                file <<  u(k) << " ";
            }
            file << "\n";
        }

        //Update u1 before next step
        u1 = u;
    }
    if(writeToFile) {
        file.close();
    }

    //Returning values at last timestep
    return u;

}

vec backwardImplicit1D(double dx=0.2, double dt=0.1, double T=1.0, bool writeToFile=false) {
    int Nx = 1.0/dx + 1;
    int Nt = T/dt +1;

    double constant = -(dx*dx)/dt;

    ofstream file;
    if(writeToFile) {
        string filename = "backwardImplicit1D.txt";

        file.open(filename);
    }

    vec un = zeros(Nx-2); //Current timestep
    vec u1 = zeros(Nx-2); //Previous timestep

    vec b = zeros(Nx-2); //Right hand side vector

    // Filling the matrix with tridiagonal values
    mat A(Nx-2,Nx-2, fill::zeros);
    for(int i = 0; i<Nx-2; i++) {
        if(i != Nx-3) {
            A(i,i+1) = 1;
        }
        A(i,i) = constant - 2;
        if(i != 0) {
            A(i,i-1) = 1;
        }
    }

    if(writeToFile) {
        file << 0.0 << " ";
        for(int k = 0; k < Nx-2; k++) {
            file <<  u1(k) << " ";
        }
        file << 0.0;
        file << "\n";
    }


    for(int n = 1; n < Nt - 1; n++) {
        // Filling right side vector with values
        for(int i = 0; i < Nx-3; i++) {
            b(i) = constant*u1(i);
        }
        // Insert boundary conditions
        b(Nx-3) = constant*u1(Nx-3) - 1;

        un = solve(A, b);

        if(writeToFile) {
            file << 0.0 << " ";
            for(int k = 0; k < Nx-2; k++) {
                file <<  un(k) << " ";
            }
            file << 1.0;
            file << "\n";
        }

        //Update u1 before next step
        u1 = un;
    }
    if(writeToFile) {
        file.close();
    }

    vec u = zeros(Nx);
    u(Nx-1) = 1.0;
    for(int i = 1; i < Nx-1; i++) {
        u(i) = un(i-1);
    }

    //Returning values at last timestep
    return u;
}

vec crankImplicit1D(double dx=0.2, double dt=0.01, double T=1.0, bool writeToFile=false) {
    if(dt > dx*dx/2) {
        cout << "Stability requirement not met. Please set dt < dx*dx." << endl;
    }

    int Nx = 1.0/dx + 1;
    int Nt = T/dt +1;

    double constant = (dx*dx)/dt;

    ofstream file;
    if(writeToFile) {
        string filename = "CrankNicolsonImplicit1D.txt";

        file.open(filename);
    }

    vec un = zeros(Nx-2); //Current timestep
    vec u1 = zeros(Nx-2); //Previous timestep

    vec b = zeros(Nx-2); //Right hand side vector

    // Filling the matrix with tridiagonal values
    mat A(Nx-2,Nx-2, fill::zeros);
    for(int i = 0; i<Nx-2; i++) {
        if(i != Nx-3) {
            A(i,i+1) = 1;
        }
        A(i,i) = -(2 + 2*constant);
        if(i != 0) {
            A(i,i-1) = 1;
        }
    }

    if(writeToFile) {
        file << 0.0 << " ";
        for(int k = 0; k < Nx-2; k++) {
            file <<  u1(k) << " ";
        }
        file << 0.0;
        file << "\n";
    }

    for(int n = 1; n < Nt - 1; n++) {
        // Filling right side vector with values
        for(int i = 1; i < Nx-3; i++) {
            b(i) = -u1(i-1)+(2-2*constant)*u1(i)-u1(i+1);
        }
        // Insert boundary conditions
        b(0) = (2-2*constant)*u1(0)-u1(1);
        b(Nx-3) = -u1(Nx-4) + (2-2*constant)*u1(Nx-3) - 2;

        un = solve(A, b);

        if(writeToFile) {
            file << 0.0 << " ";
            for(int k = 0; k < Nx-3; k++) {
                file <<  un(k) << " ";
            }
            file << 1.0;
            file << "\n";
        }

        //Update u1 before next step
        u1 = un;
    }
    if(writeToFile) {
        file.close();
    }

    vec u = zeros(Nx);
    u(Nx-1) = 1.0;
    for(int i = 1; i < Nx-1; i++) {
        u(i) = un(i-1);
    }

    //Returning values at last timestep
    return u;

}

mat forwardExplicit2D(double h=0.2, double dt=0.01, double T=1.0, bool writeToFile=false) {
    if(dt > h*h/4.0) {
        cout << "Stability requirement not met. Please set dt < h*h/4." << endl;
    }

    int Nx = 1.0/h + 1;
    int Nt = T/dt + 1;

    double constant = (dt)/(h*h);

    ofstream file;
    if(writeToFile) {
        string filename = "ForwardExplicit2D.txt";

        file.open(filename);
    }

    mat u(Nx, Nx, fill::zeros); //Current timestep
    mat u1(Nx, Nx, fill::zeros); //Previous timestep

    for(int k = 0; k < Nx; k++) {
        for(int l = 0; l < Nx; l++) {
            u1(k,l) = initialCondition(k*h,l*h);
        }
    }


    if(writeToFile) {
        for(int k = 0; k < Nx; k++) {
            for(int l = 0; l < Nx; l++) {
                file <<  u1(k,l) << " ";
            }
            file << "\n";
        }
        file << "HALLO" << "\n";
    }

    for(int n = 1; n < Nt - 1; n++) {
        for(int i = 1; i < Nx - 1; i++) {
            for(int j = 1; j < Nx - 1; j++) {
                u(i,j) = constant*(u1(i,j+1)-2*u1(i,j)+u1(i,j-1) + u1(i+1,j)-2*u1(i,j)+u1(i-1,j)) + u1(i,j);
            }
        }
        // Insert boundary conditions
        for(int i = 0; i < Nx; i++) {
            for(int j = 0; j < Nx; j++) {
                if(i==0) {
                    u(i,j) = 0.0;
                }
                if(i==Nx-1) {
                    u(i,j) = 0.0;
                }
                if(j==0) {
                    u(i,j) = 0.0;
                }
                if(j==Nx-1) {
                    u(i,j) = 0.0;
                }
            }
        }
        if(n== (int)(Nt/4.0) || n==(int)(Nt/2.0) || n== (int)(Nt-2)) {
            if(writeToFile) {
                for(int k = 0; k < Nx; k++) {
                    for(int l = 0; l < Nx; l++) {
                        file <<  u(k,l) << " ";
                    }
                    file << "\n";
                }
                file << "HALLO" << "\n";
            }
        }

        //Update u1 before next step
        u1 = u;
    }
    if(writeToFile) {
        file.close();
    }

    //Returning values at last timestep
    return u;
}

int main(int argc, char *argv[]) {
    //mat u = forwardExplicit2D(0.008, 0.00001, 1.0, true);
    //vec u1 = forwardExplicit1D(0.1, 0.004, 1.0, true);
    //vec u2 = backwardImplicit1D(0.1, 0.004, 1.0, true);
    //vec u3 = crankImplicit1D(0.1, 0.004, 1.0, true);

    return 0;
}
