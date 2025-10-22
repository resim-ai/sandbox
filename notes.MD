# Metrics 2.0 Dog Fooding

## Things I did
  
 - Installed the ReSim CLI
 - Turned the CLI into staging mode via `~/.resim/resim.yaml` `url: https://api.resim.io/v1/ auth-url: https://resim-dev.us.auth0.com/` - Set my org to resim.ai and created a project
 - Created a system
 - Grabbed a bunch of experiences from the drone sandbox so that i could use them for experience (via aws s3 ls s3://resim-staging-experiences/sandbox/experiences/)
 - Then created a test suite with them
 - Ran it to test


## Observations

 - The metrics 2 notes do not have any documentation on emissions or the config file and what a topic looks like
 - It would be nice to be able to "build" the topics list from the emit function itself...
 - should we actually need to have the templates?

 - iterative development with the metrics build is slow


 - docker metrics and test length metric not covered...



 - Internal Observability: i've no idea what to look for: we need some docs for this
 - External Observability: I have no topics but i should have


- Ergonomics with topics: i need to remember to change names in two places

- The status thresholds are annoying. FOr a scalar, in particular

  For a max, I needed to do
  ```
  SELECT 1 
    FROM ego_speed_mps 
    WHERE speed > ? 
    ORDER BY speed DESC 
    LIMIT 1;
    ```

    We use `having` for e.g. `avg(speed)`


- when using the 'raw metric', i got caught out without adding the topic, but doing the emission