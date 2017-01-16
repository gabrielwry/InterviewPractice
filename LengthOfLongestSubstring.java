/*Given a string, find the length of the longest substring without repeating characters.

Examples:

Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
*/
/*solution: keep a hashmap of all chars, a start pointer and a end pointer =>
            pointers only go foward  => if char at end position is larger than
            start, update start  => update size
*/ 
public class Solution {
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Integer> pos = new HashMap<Character, Integer>();
        int start = 0;
        int end = 0;
        int size =0;
        int max=size;
        while(end<s.length()){
           char c = s.charAt(end);
           if(!pos.containsKey(c) || pos.get(c)<start){
               size++;
           }
           else{
               if(size>max) max=size;
               start=pos.get(c)+1;
               size=end-start+1;
           }
           pos.put(c,end);
           end++;
        }
        return Math.max(size,max);
        
    }
}
