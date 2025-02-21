import { Routes } from "@angular/router";
import { NodeComponent } from "./node.component";
import { AttackComponent } from "./routes/attack/attack.component";
import { ConfigurationComponent } from "./routes/configuration/configuration.component";
import { NetworkTrafficComponent } from "./routes/network-traffic/network-traffic.component";

export const routes: Routes = [
    {
        path: "",
        component: NodeComponent,
        children: [
            { path: "network-traffic", component: NetworkTrafficComponent },
            { path: "attack", component: AttackComponent },
            { path: "configuration", component: ConfigurationComponent },
            { path: "**", redirectTo: "network-traffic" },
        ],
    },
];
