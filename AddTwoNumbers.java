/**
* You are given two non-empty linked lists representing two non-negative integers. The digits are 
* stored in reverse order and each of their nodes contain a single digit. Add the two numbers and 
* return it as a linked list.
* You may assume the two numbers do not contain any leading zero, except the number 0 itself.
* Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
* Output: 7 -> 0 -> 8
* Definition for singly-linked list.
* public class ListNode {
*     int val;
*     ListNode next;
*     ListNode(int x) { val = x; }
* }
*/
public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
       int n1=0;
       int n2=0;
       int carry =0;
       int bit=0;
       int sum=0;
       ListNode ans = new ListNode(0);
       ListNode c1=l1;
       ListNode c2=l2;
       ListNode preans=ans;
       while(c1!=null || c2!=null){
           ans.next=new ListNode(0);
           ans=ans.next;
           sum=carry;
         if(c1!=null){
            n1=c1.val;
            c1=c1.next;
            sum+=n1;
         }
         if(c2!=null){
             n2=c2.val;
             c2=c2.next;
             sum+=n2;
         }
         carry=sum/10;
         bit=sum%10;
         ans.val=bit;
       }
       if(carry!=0){
           ans.next=new ListNode(carry);
       }
       return preans.next;
        
    }
}