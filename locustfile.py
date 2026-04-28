from locust import HttpUser, between, task


class RecommenderUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def recommend(self):
        self.client.post(
            "/recommend",
            json={"user_id": 123, "top_n": 5, "exclude_rated": True},
        )

    @task(2)
    def similar(self):
        self.client.post(
            "/similar",
            json={"recipe_id": 456, "top_n": 5},
        )

    @task(1)
    def new_user(self):
        self.client.post(
            "/recommend/new-user",
            json={"liked_recipe_ids": [456], "top_n": 5},
        )

    @task(1)
    def metrics(self):
        self.client.get("/metrics")
