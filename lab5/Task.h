//
// Created by Roman on 22.05.2022.
//

#ifndef LAB5_TASK_H
#define LAB5_TASK_H


class Task {
private:
    int repeatNum;
    
public:
    Task(int repeatNum) : repeatNum(repeatNum) {
    }
    
    Task(){}
    
    void setRepeatNum(int repeat_num) {
        repeatNum = repeat_num;
    }
    
    int getRepeatNum() const {
        return repeatNum;
    }
    
    double perform() {
        double sum = 0;
        for (int i = 0; i < repeatNum; i++) {
            sum += repeatNum;
        }
        return sum;
    }
};


#endif //LAB5_TASK_H
