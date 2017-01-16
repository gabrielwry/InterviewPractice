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