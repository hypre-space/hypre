#include <stdlib.h> 
#include <stdio.h>

#include "queue.h"

//============================================================================
queue::queue(){
  elements = 0;
}

//============================================================================
void queue::push(int el){
  if (elements == 2){
    printf("Queue overflow. Exit\n");
    exit(1);
  }
  q[elements] = el;
  elements++;
}

//============================================================================
int queue::pop(){
  if (elements==0){
    printf("Queue underflow. Exit\n");
    exit(1);
  }
  int result = q[0], i;
  elements--;
  for(i=0; i<elements; i++)
    q[i] = q[i+1];
  return result;
}

//============================================================================
int queue::first(){
  return q[0];
}

//============================================================================
int queue::get(int i){
  return q[i];
}

//============================================================================
int queue::empty(){
  if (elements == 0)
    return 1;
  else return 0;
}

//============================================================================
int queue::size(){
  return elements;
}

//============================================================================
