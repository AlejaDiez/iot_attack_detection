import { Routes } from "@angular/router";
import { AttackComponent } from "./attack/attack.component";
import { ConfigurationComponent } from "./configuration/configuration.component";
import { NetworkTrafficComponent } from "./network-traffic/network-traffic.component";
import { NodeComponent } from "./node.component";

export const routes: Routes = [
    {
        path: "",
        component: NodeComponent,
        children: [
            { path: "network-traffic", component: NetworkTrafficComponent },
            { path: "attack", component: AttackComponent },
            { path: "configuration", component: ConfigurationComponent },
        ],
    },
];
