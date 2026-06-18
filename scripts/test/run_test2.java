import java.util.*;
public class run_test2 {
    public static void main(String[] args) {
        Map<String, String> plugins = new HashMap<>();
        String id = "model-family/broken-descriptor";
        String shortId = id.startsWith("model-family/") ? id.substring("model-family/".length()) : id;
        plugins.put(shortId, "plugin");
        System.out.println(plugins.get("broken-descriptor"));
    }
}
