public class run_test4 {
    public static void main(String[] args) {
        String id = "model-family/broken-descriptor";
        String shortId = id.startsWith("model-family/") ? id.substring("model-family/".length()) : id;
        System.out.println(shortId);
    }
}
