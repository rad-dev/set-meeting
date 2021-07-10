package meeting;

import java.util.ArrayList;
import java.util.InvalidPropertiesFormatException;
import java.util.List;

public class UserBase {
   private List<Person> users;
   private  List<String> userEmails;

   public UserBase() {
       users = new ArrayList<>();
       userEmails = new ArrayList<>();
   }

   public void addUser(Person user) throws InvalidPropertiesFormatException {
       if (!userEmails.contains(user.getEmail())) {
           users.add(user);
           userEmails.add(user.getEmail());
       } else {
           throw new InvalidPropertiesFormatException("Email already exists!");
       }
   }

   public Person getUser(String email) {
       return  users.stream().filter(user -> email.equals(user.getEmail())).findFirst().orElse(null);
   }
}
